"""
ComfyUI Nodes: Anima LoRA dataset preparation

Three nodes that turn a folder of raw images into a dataset the trainer accepts:

  - AnimaLoadImages:    read a folder (recursive optional), pair .txt captions
  - AnimaResizeImages:  scale each image to a training bucket around a target area
  - AnimaSaveDataset:   write <repeats>_<concept>/<image>.png + .txt for train_data_dir

The save node produces the kohya structure that check_dataset() requires, so its
train_data_dir output can be wired straight into Basic Parameters / the Wizard.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import folder_paths
import node_helpers

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _resolve_folder(folder: str) -> str:
    folder = (folder or "").strip()
    if not folder:
        raise ValueError("Folder is empty: set a path, or a folder name relative to the ComfyUI input directory.")
    if not os.path.isabs(folder):
        folder = os.path.join(folder_paths.get_input_directory(), folder)
    path = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(path):
        raise ValueError(f"Folder does not exist: {path}")
    return path


def _list_images(root: str, recursive: bool) -> list:
    if not recursive:
        names = sorted(os.listdir(root))
        return [
            os.path.join(root, name)
            for name in names
            if name.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(root, name))
        ]

    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        paths.extend(
            os.path.join(dirpath, name)
            for name in sorted(filenames)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        )
    return paths


def _read_caption(image_path: str) -> str:
    caption_path = os.path.splitext(image_path)[0] + ".txt"
    if not os.path.isfile(caption_path):
        return ""
    with open(caption_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _load_image(image_path: str, max_side: int) -> torch.Tensor:
    image = node_helpers.pillow(Image.open, image_path)
    image = ImageOps.exif_transpose(image)
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    if image.mode == "I":
        image = image.point(lambda i: i * (1 / 255))
    pixels = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(pixels)[None]


def _plan_bucket(width, height, base_resolution, step, max_side, allow_upscale):
    """Area-targeted bucket, aspect ratio kept, both sides snapped to a multiple of step."""
    scale = ((base_resolution * base_resolution) / (width * height)) ** 0.5
    if not allow_upscale:
        scale = min(scale, 1.0)

    width *= scale
    height *= scale
    if max(width, height) > max_side:
        shrink = max_side / max(width, height)
        width *= shrink
        height *= shrink

    if allow_upscale:
        return max(step, round(width / step) * step), max(step, round(height / step) * step)
    # Rounding up would grow the source again, so floor to stay at or below its own size.
    return max(step, int(width // step) * step), max(step, int(height // step) * step)


def _resample(image, width, height):
    x = image.movedim(-1, 1)
    if width * height < x.shape[2] * x.shape[3]:
        x = F.interpolate(x, size=(height, width), mode="area")
    else:
        x = F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False)
    return x.movedim(1, -1)


def _fit(image, width, height, fit):
    if fit == "center_crop":
        src_h, src_w = image.shape[1], image.shape[2]
        scale = max(width / src_w, height / src_h)
        scaled_w, scaled_h = round(src_w * scale), round(src_h * scale)
        image = _resample(image, max(width, scaled_w), max(height, scaled_h))
        top = (image.shape[1] - height) // 2
        left = (image.shape[2] - width) // 2
        return image[:, top:top + height, left:left + width]
    return _resample(image, width, height)


def _to_pil(image):
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected one image per item, got shape {tuple(image.shape)}")
    pixels = image[0].clamp(0, 1).cpu().numpy()
    pixels = (pixels * 255.0).round().astype(np.uint8)
    if pixels.shape[-1] == 4:
        pixels = pixels[..., :3]
    return Image.fromarray(pixels)


def _clean_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename((name or "").replace("\\", "/")))[0]
    stem = "".join(c for c in stem if c.isalnum() or c in " -_.")
    return stem.strip(" .")


class AnimaLoadImages:
    """
    Read every image in a folder and pair it with the .txt caption that sits next to it.
    Outputs lists so the rest of the dataset nodes can work on the whole batch at once.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Absolute path, or a path relative to ComfyUI/input."
                }),
                "recursive": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Also walk subfolders."
                }),
                "max_images": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "0 loads every image; otherwise the first N in name order."
                }),
                "max_side": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 64,
                    "tooltip": "Downscale while decoding when a side is larger. 0 keeps the original size."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT")
    RETURN_NAMES = ("images", "captions", "filenames", "count")
    OUTPUT_IS_LIST = (True, True, True, False)
    FUNCTION = "execute"
    CATEGORY = "Anima/LoRA/Data"

    def execute(self, folder, recursive, max_images, max_side):
        root = _resolve_folder(folder)
        paths = _list_images(root, recursive)
        if max_images > 0:
            paths = paths[:max_images]
        if not paths:
            raise ValueError(f"No images ({', '.join(IMAGE_EXTENSIONS)}) found in: {root}")

        images = [_load_image(path, max_side) for path in paths]
        captions = [_read_caption(path) for path in paths]
        filenames = [os.path.relpath(path, root).replace("\\", "/") for path in paths]
        return (images, captions, filenames, len(images))


class AnimaResizeImages:
    """
    Scale every image to a training bucket around a target pixel area.

    base_resolution is an area (base_resolution squared), not a side length: the image
    is scaled to that area with its aspect ratio kept, then both sides are snapped to a
    multiple of bucket_step. That is the same bucket math the trainer applies, so it
    will not resize your images a second time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Image list, usually from Load Images From Folder."
                }),
                "base_resolution": ("INT", {
                    "default": 1024, "min": 64, "max": 4096, "step": 64,
                    "tooltip": "Bucket area is base_resolution squared. 1024 gives ~1024x1024 worth of pixels."
                }),
                "bucket_step": ("INT", {
                    "default": 64, "min": 8, "max": 256, "step": 8,
                    "tooltip": "Bucket granularity. sd-scripts / kohya use 64."
                }),
                "max_side": ("INT", {
                    "default": 2048, "min": 256, "max": 8192, "step": 64,
                    "tooltip": "Hard cap for the long side, matching max_bucket_reso."
                }),
                "fit": (["snap", "center_crop"], {
                    "default": "snap",
                    "tooltip": "snap squeezes onto the bucket (under one step of distortion); "
                               "center_crop resizes to cover and crops the overflow."
                }),
                "allow_upscale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Let images below base_resolution grow to their bucket."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "sizes")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = "Anima/LoRA/Data"

    def execute(self, images, base_resolution, bucket_step, max_side, fit, allow_upscale):
        # INPUT_IS_LIST wraps every input in a list, widget values included.
        base_resolution = base_resolution[0]
        bucket_step = bucket_step[0]
        max_side = max_side[0]
        fit = fit[0]
        allow_upscale = allow_upscale[0]

        resized = []
        sizes = []
        for image in images:
            items = image.split(1, dim=0) if image.shape[0] > 1 else (image,)
            for item in items:
                height, width = item.shape[1], item.shape[2]
                bucket_w, bucket_h = _plan_bucket(width, height, base_resolution, bucket_step, max_side, allow_upscale)
                resized.append(_fit(item, bucket_w, bucket_h, fit))
                sizes.append(f"{bucket_w}x{bucket_h}")
        return (resized, sizes)


class AnimaSaveDataset:
    """
    Write the images into <folder>/<repeats>_<concept>/ as PNG plus a .txt caption next
    to each one. That is the layout check_dataset() and sd-scripts expect, so the
    train_data_dir output can go straight into Basic Parameters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Parent folder. Images go into <folder>/<repeats>_<concept>/."
                }),
                "images": ("IMAGE", {
                    "tooltip": "Image list to write."
                }),
                "repeats": ("INT", {
                    "default": 1, "min": 1, "max": 1000, "step": 1,
                    "tooltip": "How many times each image is seen per epoch (the <repeats>_ prefix)."
                }),
                "concept": ("STRING", {
                    "default": "train",
                    "multiline": False,
                    "tooltip": "Subfolder name after the repeats prefix, e.g. 'firefly' -> '10_firefly'."
                }),
            },
            "optional": {
                "captions": ("STRING", {
                    "tooltip": "Caption per image; empty ones are skipped."
                }),
                "filenames": ("STRING", {
                    "tooltip": "Names to reuse, e.g. the filenames output of Load Images From Folder."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("paths", "count", "train_data_dir")
    OUTPUT_IS_LIST = (True, False, False)
    INPUT_IS_LIST = True
    FUNCTION = "execute"
    CATEGORY = "Anima/LoRA/Data"

    def execute(self, folder, images, repeats, concept, captions=None, filenames=None):
        # INPUT_IS_LIST wraps every input in a list, widget values included.
        parent = folder[0].strip()
        repeats = repeats[0]
        concept = concept[0].strip()

        if not parent:
            raise ValueError("Folder is empty: set where the dataset should be written.")
        parent = os.path.abspath(os.path.expanduser(parent))

        name = f"{repeats}_{concept}" if concept else str(repeats)
        target = os.path.join(parent, name)
        os.makedirs(target, exist_ok=True)

        filenames = filenames or []
        captions = captions or []

        paths = []
        used = set()
        for index, image in enumerate(images):
            stem = _clean_stem(filenames[index]) if index < len(filenames) else ""
            stem = stem or f"{index + 1:05d}"

            unique = stem
            suffix = 2
            while unique.lower() in used:
                unique = f"{stem}_{suffix}"
                suffix += 1
            used.add(unique.lower())

            image_path = os.path.join(target, unique + ".png")
            _to_pil(image).save(image_path)

            caption = captions[index].strip() if index < len(captions) and captions[index] else ""
            if caption:
                with open(os.path.join(target, unique + ".txt"), "w", encoding="utf-8") as f:
                    f.write(caption)

            paths.append(image_path)

        return (paths, len(paths), parent)


NODE_CLASS_MAPPINGS = {
    "AnimaLoadImages": AnimaLoadImages,
    "AnimaResizeImages": AnimaResizeImages,
    "AnimaSaveDataset": AnimaSaveDataset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoadImages": "Load Images From Folder (Anima)",
    "AnimaResizeImages": "Resize Images For Training (Anima)",
    "AnimaSaveDataset": "Save Dataset (Anima)",
}
