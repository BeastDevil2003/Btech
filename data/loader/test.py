from data.loader.ffpp_dataset_2 import FFPPDataset
from data.loader.build_dataset import build_ffpp_dataset
from torch.utils.data import DataLoader

root = "FaceForensics_Data"

video_paths, labels, mask_paths, domains = build_ffpp_dataset(root)

# small test
video_paths = video_paths[:8]
labels = labels[:8]
mask_paths = mask_paths[:8]
domains = domains[:8]

dataset = FFPPDataset(video_paths, labels, mask_paths, domains)
loader = DataLoader(dataset, batch_size=2)

batch = next(iter(loader))

print("Frames:", batch["frames"].shape)
print("Masks:", batch["mask_frames"].shape)
print("Labels:", batch["label"])
print("Has mask:", batch["has_mask"])
print("Domain:", batch["domain"])

