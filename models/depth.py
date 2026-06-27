
import torch, cv2, numpy as np
 
class DepthEstimator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MiDaS] Using device: {self.device}")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device).eval()
        transforms    = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        print("[MiDaS] ✅ Model loaded")
 
    def estimate(self, image: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inp     = self.transform(img_rgb).to(self.device)
        with torch.no_grad():
            depth = self.model(inp)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1), size=image.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze()
        depth_np  = depth.cpu().numpy()
        depth_ref = cv2.bilateralFilter(depth_np.astype(np.float32), 9, 75, 75)
        return cv2.normalize(depth_ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
 
    def colorize(self, depth_map: np.ndarray) -> np.ndarray:
        return cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
