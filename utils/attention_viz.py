import torch
import cv2
import os
import numpy as np

def save_attention_map(model, image, epoch, save_dir="outputs/attention"):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        out = model(image, return_attention=True)
        attn = out["attention"]  

    attn = attn[0].cpu().numpy()
    attn = cv2.resize(attn, (224, 224))
    attn = (attn - attn.min()) / (attn.max() - attn.min())

    heatmap = cv2.applyColorMap(
        np.uint8(255 * attn),
        cv2.COLORMAP_JET
    )

    img = image[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(
        f"{save_dir}/epoch_{epoch}.png",
        overlay
    )
