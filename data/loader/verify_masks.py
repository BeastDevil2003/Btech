import matplotlib.pyplot as plt
from data.loader.ffpp_dataset_2 import FFPPDataset
from data.loader.build_dataset import build_ffpp_dataset


def visualize_sample(dataset):
    sample = None

    # Find sample with mask
    for i in range(50):    #len(dataset)
        s = dataset[i]
        if s["has_mask"]:
            sample = s
            print(f"Using sample index: {i}")
            break

    if sample is None:
        print("❌ No masked sample found!")
        return

    frames = sample["frames"]       # (N, 6, H, W)
    masks = sample["mask_frames"]   # (N, H, W)

    # Convert first frame (RGB only)
    frame = frames[0][:3]  # take RGB
    frame = frame.permute(1, 2, 0).numpy()

    # Undo normalization (since we did (x-0.5)/0.5)
    frame = (frame * 0.5) + 0.5

    mask = masks[0].numpy()

    # -----------------------------
    # SINGLE FRAME VIEW
    # -----------------------------
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Frame")
    plt.imshow(frame)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.show()

    # -----------------------------
    # OVERLAY VIEW
    # -----------------------------
    plt.figure(figsize=(5, 5))
    plt.title("Overlay (Frame + Mask)")
    plt.imshow(frame)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

    # -----------------------------
    # TEMPORAL CHECK (first 5 frames)
    # -----------------------------
    plt.figure(figsize=(12, 5))

    for i in range(5):
        # Frame
        f = frames[i][:3].permute(1, 2, 0).numpy()
        f = (f * 0.5) + 0.5

        # Mask
        m = masks[i].numpy()

        plt.subplot(2, 5, i + 1)
        plt.imshow(f)
        plt.title(f"Frame {i}")
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(m, cmap='gray')
        plt.title(f"Mask {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root = "FaceForensics_Data"  # 🔥 change if needed

    video_paths, labels, mask_paths, domains = build_ffpp_dataset(root)

    dataset = FFPPDataset(video_paths, labels, mask_paths, domains)

    visualize_sample(dataset)