"""
training/augmentation.py
MeterAugmentor — all required augmentations for meter digit training.
Usable as a torchvision-compatible transform or standalone.
"""
import random
import math
import numpy as np
import cv2


def _glare_patch(image: np.ndarray) -> np.ndarray:
    """Simulate a specular glare patch on the image."""
    h, w = image.shape[:2]
    n_patches = random.randint(1, 3)
    out = image.copy()
    for _ in range(n_patches):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        rx = random.randint(15, max(16, w // 4))
        ry = random.randint(8, max(9, h // 4))
        intensity = random.uniform(0.4, 0.9)
        mask = np.zeros((h, w), np.float32)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        out = np.clip(out.astype(np.float32) + mask[:, :, None] * 255 * intensity, 0, 255).astype(np.uint8)
    return out


def _perspective_warp(image: np.ndarray, strength: float = 0.08) -> np.ndarray:
    h, w = image.shape[:2]
    d = lambda: random.uniform(-strength, strength)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * d(), h * d()],
        [w*(1+d()), h * d()],
        [w*(1+d()), h*(1+d())],
        [w * d(), h*(1+d())]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)


def _rotate(image: np.ndarray, max_angle: float = 30.0) -> np.ndarray:
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def _decimal_shift(image: np.ndarray, max_px: int = 5) -> np.ndarray:
    """Shift image slightly to simulate decimal misalignment."""
    dx = random.randint(-max_px, max_px)
    dy = random.randint(-max_px, max_px)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = image.shape[:2]
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


class MeterAugmentor:
    """
    Heavy augmentation pipeline for meter digit / decimal crops.

    All augmentations are randomly applied with the given probabilities.
    """

    def __init__(
        self,
        brightness_range: float = 0.40,
        contrast_range: float   = 0.40,
        blur_max: float         = 2.0,
        rotation_max: float     = 30.0,
        perspective_strength: float = 0.08,
        decimal_shift_px: int   = 5,
        glare_prob: float       = 0.30,
        p_augment: float        = 0.85,
    ):
        self.brightness_range     = brightness_range
        self.contrast_range       = contrast_range
        self.blur_max             = blur_max
        self.rotation_max         = rotation_max
        self.perspective_strength = perspective_strength
        self.decimal_shift_px     = decimal_shift_px
        self.glare_prob           = glare_prob
        self.p_augment            = p_augment

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p_augment:
            return image

        img = image.copy().astype(np.float32)

        # Brightness ±40%
        if random.random() < 0.7:
            bf = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            img = np.clip(img * bf, 0, 255)

        # Contrast ±40%
        if random.random() < 0.7:
            cf = 1.0 + random.uniform(-self.contrast_range, self.contrast_range)
            mean = img.mean()
            img  = np.clip(mean + (img - mean) * cf, 0, 255)

        img = img.astype(np.uint8)

        # Gaussian blur 0–2.0 px
        if random.random() < 0.5:
            sigma = random.uniform(0, self.blur_max)
            ks    = max(1, int(sigma * 2) | 1)  # odd kernel
            img   = cv2.GaussianBlur(img, (ks, ks), sigma)

        # Rotation ±30°
        if random.random() < 0.6:
            img = _rotate(img, self.rotation_max)

        # Perspective warp
        if random.random() < 0.5:
            img = _perspective_warp(img, self.perspective_strength)

        # Decimal shift ±5px
        if random.random() < 0.4:
            img = _decimal_shift(img, self.decimal_shift_px)

        # Glare patch
        if random.random() < self.glare_prob:
            img = _glare_patch(img)

        return img

    def augment_batch(self, images: list) -> list:
        return [self(img) for img in images]
