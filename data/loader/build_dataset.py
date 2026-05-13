import os


def build_ffpp_dataset(root_dir, compression="c23"):   #  for c40 give value here  or raw for raw
    video_paths = []
    labels = []
    mask_paths = []
    domains = []

    manipulated_root = os.path.join(root_dir, "manipulated_sequences")
    original_root = os.path.join(root_dir, "original_sequences")

    # -------------------------
    # REAL VIDEOS (YouTube + Actors)
    # -------------------------
    youtube_dir = os.path.join(original_root, "youtube")
    actors_dir = os.path.join(original_root, "actors")

    # YouTube
    for root, _, files in os.walk(youtube_dir):
        for f in files:
            if f.endswith(".mp4"):
                video_paths.append(os.path.join(root, f))
                labels.append(0)
                mask_paths.append(None)
                domains.append("youtube")

    # Actors
    for root, _, files in os.walk(actors_dir):
        for f in files:
            if f.endswith(".mp4"):
                video_paths.append(os.path.join(root, f))
                labels.append(0)
                mask_paths.append(None)
                domains.append("actors")

    # -------------------------
    # FAKE VIDEOS
    # -------------------------
    fake_types = os.listdir(manipulated_root)

    for fake_type in fake_types:
        fake_path = os.path.join(manipulated_root, fake_type)

        video_dir = os.path.join(fake_path, compression)
        mask_dir = os.path.join(fake_path, "masks")

        for root, _, files in os.walk(video_dir):
            for f in files:
                if f.endswith(".mp4"):
                    video_paths.append(os.path.join(root, f))
                    labels.append(1)
                    domains.append(fake_type)

                    # FaceShifter has no masks
                    if fake_type == "FaceShifter":
                        mask_paths.append(None)
                    else:
                        # Mask video path
                        mask_path = os.path.join(mask_dir, "videos", f)

                        if os.path.exists(mask_path):
                            mask_paths.append(mask_path)
                        else:
                            mask_paths.append(None)

    return video_paths, labels, mask_paths, domains