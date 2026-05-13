import os
import cv2


def inspect_ffpp_dataset(root_dir):
    manipulated_root = os.path.join(root_dir, "manipulated_sequences")
    original_root = os.path.join(root_dir, "original_sequences")

    print("\n[INFO] DATASET INSPECTION STARTED\n")

    # -------------------------
    # REAL VIDEOS (DETAILED)
    # -------------------------
    youtube_dir = os.path.join(original_root, "youtube")
    actors_dir = os.path.join(original_root, "actors")

    youtube_count = 0
    actors_count = 0

    # youtube
    for root, _, files in os.walk(youtube_dir):
        for f in files:
            if f.endswith(".mp4"):
                youtube_count += 1

    # actors
    for root, _, files in os.walk(actors_dir):
        for f in files:
            if f.endswith(".mp4"):
                actors_count += 1

    real_count = youtube_count + actors_count

    print(f"[OK] Real Videos Total: {real_count}")
    print(f"   YouTube: {youtube_count}")
    print(f"   Actors: {actors_count}")

    # -------------------------
    # FAKE VIDEOS
    # -------------------------
    total_fake = 0

    for fake_type in os.listdir(manipulated_root):
        fake_path = os.path.join(manipulated_root, fake_type)

        video_dir = os.path.join(fake_path, "c23")
        mask_dir = os.path.join(fake_path, "masks")

        print(f"\n[FOLDER] {fake_type}")

        # ---- Video count ----
        video_files = []
        for root, _, files in os.walk(video_dir):
            for f in files:
                if f.endswith(".mp4"):
                    video_files.append(f)

        print(f"   Videos: {len(video_files)}")
        total_fake += len(video_files)

        # ---- Mask analysis ----
        if not os.path.exists(mask_dir):
            print("   [MISSING] No mask folder")
            continue

        mask_files = os.listdir(mask_dir)

        # Detect mask type
        if "videos" in mask_files:
            print("   [MASK] Type: VIDEO")
            mask_video_dir = os.path.join(mask_dir, "videos")

            mask_video_files = [f for f in os.listdir(mask_video_dir) if f.endswith(".mp4")]

            print(f"   Mask Videos: {len(mask_video_files)}")

            # Check matching
            match_count = len(set(video_files) & set(mask_video_files))
            print(f"   Matched Videos: {match_count}")

        else:
            # Check file types
            sample = mask_files[:5]

            print(f"   Mask Files (sample): {sample}")

            png_count = sum(1 for f in mask_files if f.endswith(".png"))
            mp4_count = sum(1 for f in mask_files if f.endswith(".mp4"))

            if png_count > 0:
                print("   [MASK] Type: IMAGE (.png)")
            elif mp4_count > 0:
                print("   [MASK] Type: VIDEO (.mp4)")
            else:
                print("   [WARN] Unknown mask format")

    # -------------------------
    # FINAL SUMMARY
    # -------------------------
    print("\n[SUMMARY] FINAL SUMMARY")
    print("------------------------")
    print(f"Total Real Videos: {real_count}")
    print(f"Total Fake Videos: {total_fake}")

    print("\n[INFO] INSPECTION DONE\n")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    root = "FaceForensics_Data"  # change if needed
    inspect_ffpp_dataset(root)
