import os
import torch
import numpy as np
from PIL import Image, ImageOps
import random

class LoadRandomImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "include_subfolders": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "image_path")
    FUNCTION = "load_random_image"
    CATEGORY = "zveroboy"

    @classmethod
    def IS_CHANGED(s, folder, include_subfolders=False):
        # Возвращаем случайное значение, чтобы ComfyUI считал ноду изменённой при каждом запуске
        return random.random()

    def load_random_image(self, folder, include_subfolders=False):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' cannot be found.")
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        image_paths = []
        
        if include_subfolders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    image_paths.append(os.path.join(folder, file))

        if not image_paths:
            raise FileNotFoundError(f"No valid images found in directory '{folder}'.")

        random_image_path = random.choice(image_paths)
        
        img = Image.open(random_image_path)
        img = ImageOps.exif_transpose(img)
        
        image = img.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        height, width = image.shape[1:3]
        if 'A' in img.getbands():
            mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")

        return (image, mask, random_image_path)

NODE_CLASS_MAPPINGS = {
    "LoadRandomImage": LoadRandomImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadRandomImage": "Load Random Image"
}
