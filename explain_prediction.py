import torch
import torch.nn.functional as F
from model.model import DeepFakeDetectionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def explain(image_tensor, model):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(image_tensor, return_attention=True)

    logits = out["logits"]
    probs = F.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred].item()

    spatial_weight = out["fusion_spatial_att"].mean().item()
    freq_weight = out["fusion_frequency_att"].mean().item()

    explanation = ""

    if pred == 1:
        explanation += "Image classified as FAKE.\n"
        if freq_weight > spatial_weight:
            explanation += "Reason: Strong frequency artifacts detected (GAN traces).\n"
        else:
            explanation += "Reason: Visual inconsistencies detected in facial regions.\n"
    else:
        explanation += "Image classified as REAL.\n"
        explanation += "Reason: Natural spatial and frequency patterns observed.\n"

    explanation += f"Confidence: {confidence:.2f}"

    return explanation
