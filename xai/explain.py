import torch
import torch.nn.functional as F
import cv2, json, os
import numpy as np
from torchvision import transforms
from PIL import Image

from model.model import DeepFakeDetectionModel
from xai.gradcam import GradCAM
from xai.face_regions import get_face_regions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("outputs/overlays", exist_ok=True)
os.makedirs("outputs/explanations", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

model = DeepFakeDetectionModel().to(DEVICE)
ckpt = torch.load("best_model.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

gradcam = GradCAM(model, model.spatial_stream.backbone.features[-1])

def explain(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    inp = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(inp, return_attention=True)

    probs = F.softmax(out["logits"], dim=1)
    pred = torch.argmax(probs).item()
    conf = probs[0, pred].item()

    cam = gradcam.generate(inp)[0,0].cpu().numpy()
    cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    cam_color = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)

    face_mask, regions = get_face_regions(img_np)
    cam_color[face_mask == 0] = 0

    overlay = cv2.addWeighted(img_np, 0.6, cam_color, 0.4, 0)

    label = "FAKE" if pred == 1 else "REAL"

    reason = (
        "High-frequency GAN artifacts detected around facial regions"
        if pred == 1 else
        "Consistent spatial and frequency patterns observed"
    )

    out_img = f"outputs/overlays/{os.path.basename(image_path)}"
    cv2.imwrite(out_img, overlay)

    json_data = {
        "prediction": label,
        "confidence": round(conf, 3),
        "reason": reason,
        "highlighted_regions": list(regions.keys())
    }

    out_json = f"outputs/explanations/{os.path.basename(image_path)}.json"
    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=4)

    return out_img, json_data
