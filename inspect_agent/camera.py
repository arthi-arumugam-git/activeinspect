"""Simulated inspection camera over a procedurally generated part surface.

A Scene is a fixed physical part (texture + optional defect). capture() renders
it through camera parameters (exposure, focus, zoom ROI) under a site lighting
condition. Bad conditions (dim light, defocus, glare) degrade the default
capture; the agent can change parameters and re-capture (active perception).
Replace with a real camera driver (GigE/USB/RTSP) by implementing capture().
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2

SIZE = 256
CONDITIONS = ["nominal", "dim", "defocus", "glare", "dim+defocus", "glare+defocus", "very_dark"]
DEFECTS = ["none", "scratch", "pit"]


@dataclass
class Scene:
    seed: int
    defect: str
    condition: str
    bbox: tuple | None = None          # ground-truth defect box (x, y, w, h) in full-frame px
    _albedo: np.ndarray = field(default=None, repr=False)

    def glare_center(self) -> tuple[int, int]:
        """Glare sits on the defect when there is one (worst case), else somewhere central."""
        if self.bbox:
            x, y, w, h = self.bbox
            return x + w // 2 + 10, y + h // 2 - 8
        return 90 + (self.seed * 37) % 80, 90 + (self.seed * 91) % 80

    @staticmethod
    def make(seed: int, defect: str | None = None, condition: str | None = None) -> "Scene":
        rng = np.random.default_rng(seed)
        defect = defect or DEFECTS[rng.integers(len(DEFECTS))]
        condition = condition or CONDITIONS[rng.integers(len(CONDITIONS))]
        base = rng.normal(0.62, 0.035, (SIZE, SIZE)).astype(np.float32)
        base = cv2.GaussianBlur(base, (0, 0), 0.7)
        gx = np.linspace(-0.05, 0.05, SIZE, dtype=np.float32)
        base += gx[None, :] * rng.uniform(-1, 1)
        bbox = None
        if defect == "scratch":
            x0, y0 = rng.integers(40, 120, 2); L = rng.integers(60, 110); a = rng.uniform(0, np.pi)
            x1, y1 = int(x0 + L * np.cos(a)), int(np.clip(y0 + L * np.sin(a), 10, SIZE - 10))
            x1 = int(np.clip(x1, 10, SIZE - 10))
            mask = np.zeros_like(base); cv2.line(mask, (int(x0), int(y0)), (x1, y1), 1.0, 2, cv2.LINE_AA)
            base -= mask * rng.uniform(0.03, 0.12)
            bbox = cv2.boundingRect(np.array([[x0, y0], [x1, y1]], np.int32))
        elif defect == "pit":
            cx, cy = rng.integers(40, SIZE - 40, 2); r = int(rng.integers(3, 6))
            mask = np.zeros_like(base); cv2.circle(mask, (int(cx), int(cy)), r, 1.0, -1, cv2.LINE_AA)
            base -= cv2.GaussianBlur(mask, (0, 0), 1.0) * rng.uniform(0.04, 0.14)
            bbox = (int(cx - r), int(cy - r), 2 * r, 2 * r)
        return Scene(seed, defect, condition, bbox, np.clip(base, 0, 1))


@dataclass
class CameraParams:
    exposure: float = 1.0      # multiplicative gain
    focus: float = 0.0         # 0 = autofocus default; agent can set +1 to correct defocus
    roi: tuple | None = None   # (x, y, w, h) digital/optical zoom window
    polarizer: bool = False    # cross-polarizer filter suppresses specular glare


def capture(scene: Scene, p: CameraParams, shot: int = 0) -> np.ndarray:
    """Physically-flavoured sensor model: exposure scales signal, shot noise ~ sqrt(signal),
    constant read noise, 8-bit quantisation and clipping."""
    rng = np.random.default_rng(scene.seed * 1000 + shot)
    conds = set(scene.condition.split("+"))
    light = 0.02 if "very_dark" in conds else 0.13 if "dim" in conds else 1.0
    alb = scene._albedo
    gx = gy = None
    if p.roi:  # optical zoom: more sensor pixels on the region, noise stays per-pixel
        x, y, w, h = p.roi
        alb = cv2.resize(alb[y:y + h, x:x + w], (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
        gx, gy = (x, w), (y, h)
    img = alb * light * p.exposure
    if "glare" in conds and not p.polarizer:
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
        if gx:
            xx, yy = gx[0] + xx * gx[1] / SIZE, gy[0] + yy * gy[1] / SIZE
        cx, cy = scene.glare_center()
        img = img + 1.1 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 45.0 ** 2))).astype(np.float32)
    blur = 4.5 if "defocus" in conds else 0.0
    if p.roi:
        blur *= SIZE / p.roi[2]
    blur = blur * (1 - min(1.0, p.focus))
    if blur > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    shot_sigma = np.sqrt(np.clip(img, 0, None) * 0.0015)
    img = img + rng.normal(0, 1, img.shape).astype(np.float32) * shot_sigma + rng.normal(0, 0.012, img.shape).astype(np.float32)
    return np.clip(img * 255, 0, 255).astype(np.uint8)
