import cv2
import numpy as np
import torch
import warnings

class RealESRGANWrapper:
    """Wrapper for Real-ESRGAN super-resolution model."""
    def __init__(self, scale=4, fp16=True, model_path=None):
        self.scale = scale
        self.fp16 = fp16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            # Default to realesr-general-x4v3
            self.upsampler = RealESRGANer(
                scale=4, 
                model_path=model_path if model_path else 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth', 
                model=model, 
                tile=400, 
                tile_pad=10, 
                pre_pad=0, 
                half=fp16
            )
        except ImportError:
            # warnings.warn("Real-ESRGAN not installed. Falling back to Lanczos interpolation. `pip install basicsr realesrgan`")
            self.upsampler = None

    def enhance(self, image: np.ndarray):
        """Applies SR to the image or Lanczos fallback."""
        if self.upsampler is None:
            h, w = image.shape[:2]
            return cv2.resize(image, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LANCZOS4)
        
        try:
            output, _ = self.upsampler.enhance(image, outscale=self.scale)
            return output
        except Exception as e:
            print(f"SR Failed: {e}. Falling back to Lanczos.")
            h, w = image.shape[:2]
            return cv2.resize(image, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LANCZOS4)
