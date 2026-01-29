import cv2
import numpy as np
import matplotlib.pyplot as plt


def overlay_heatmap(image, heatmap):
    heatmap = heatmap.squeeze().cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    overlay = 0.6 * image + 0.4 * heatmap / 255.0
    return overlay


def show(title, img):
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()
