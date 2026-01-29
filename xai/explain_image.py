import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model.model import DeepFakeDetectionModel
from xai.gradcam import GradCAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = DeepFakeDetectionModel()
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE)["model_state"])
model.eval().to(DEVICE)

target_layer = model.spatial_stream.backbone.features[-1]
gradcam = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def explain_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(input_tensor, return_attention=True)

    logits = out["logits"]
    probs = F.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred].item()

    label = "FAKE ❌" if pred == 1 else "REAL ✅"

    cam = gradcam.generate(input_tensor)[0, 0].cpu().numpy()
    cam = cv2.resize(cam, img.size)

    cam_color = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)

    img_np = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.6, cam_color, 0.4, 0)

    spatial_w = out["fusion_spatial_att"].mean().item()
    freq_w = out["fusion_frequency_att"].mean().item()

    if pred == 1:
        if freq_w > spatial_w:
            reason = (
                "The image is classified as FAKE because abnormal "
                "high-frequency artifacts were detected, which are "
                "commonly introduced by GAN-based face generation."
            )
        else:
            reason = (
                "The image is classified as FAKE due to visible "
                "texture inconsistencies in key facial regions "
                "such as eyes, mouth, and facial boundaries."
            )
    else:
        reason = (
            "The image is classified as REAL because facial textures "
            "and frequency patterns appear natural and consistent."
        )

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"{label} | Confidence: {confidence:.2f}")
    plt.show()

    print("Prediction:", label)
    print("Confidence:", round(confidence, 2))
    print("Explanation:", reason)

if __name__ == "__main__":
    explain_image("test.jpg")
