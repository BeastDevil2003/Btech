import torch
from model.model import DeepFakeDetectionModel
from xai.gradcam import GradCAM
from xai.utils import overlay_heatmap, show


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = DeepFakeDetectionModel()
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval().to(DEVICE)

target_layer = model.spatial_stream.backbone.features[-1]

gradcam = GradCAM(model, target_layer)


@torch.no_grad()
def visualize_attention(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    cam = gradcam.generate(image_tensor)
    cam_img = overlay_heatmap(image_tensor, cam)
    show("Grad-CAM (Spatial Stream)", cam_img)

    out = model(image_tensor, return_attention=True)

    sa = out["spatial_attention"][0, 0].cpu()
    ca = out["channel_attention"].mean(dim=1).squeeze().cpu()

    show("CBAM Spatial Attention", sa)
    show("CBAM Channel Attention", ca)

"""
if __name__ == "__main__":
    dummy = torch.randn(3, 224, 224)
    visualize_attention(dummy)
"""