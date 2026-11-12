"""Depth Anything V2 metric 版：輸出以公尺為單位的深度圖。"""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class MetricDepth:
    def __init__(self, scene="outdoor", size="Small", device=None):
        name = f"depth-anything/Depth-Anything-V2-Metric-{scene.capitalize()}-{size}-hf"
        self.device = device or pick_device()
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: HxWx3 uint8 → HxW float32 深度（公尺）。"""
        inputs = self.proc(images=rgb, return_tensors="pt").to(self.device)
        pred = self.model(**inputs).predicted_depth[:, None]
        pred = F.interpolate(pred, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        return pred[0, 0].float().cpu().numpy()
