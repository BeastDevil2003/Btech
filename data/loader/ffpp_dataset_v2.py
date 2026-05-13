import os
import warnings
import hashlib
import threading
import cv2
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

import insightface
from insightface.app import FaceAnalysis

from facenet_pytorch import MTCNN
from decord import VideoReader, cpu

#augmentation import
try:
    from data.augmentation import VideoAugmentation, SBIAugmentation
    _AUG_AVAILABLE = True
except ImportError:
    _AUG_AVAILABLE = False

np.seterr(all='ignore')
warnings.filterwarnings("ignore", message=".*rcond parameter will change.*", category=FutureWarning)


_WORKER_DATASET  = None
_WORKER_DETECTOR = None

def _build_cache_index_worker(dataset_kwargs, idx):
    global _WORKER_DETECTOR, _WORKER_DATASET
    if _WORKER_DATASET is None:
        _WORKER_DATASET = FFPPDatasetV2(**dataset_kwargs)
    if _WORKER_DETECTOR is None:
        _WORKER_DATASET.init_face_detector()
        _WORKER_DETECTOR = _WORKER_DATASET.face_detector
    cache_path = _WORKER_DATASET.get_cache_path(idx)
    sample = _WORKER_DATASET.build_sample(idx)
    _WORKER_DATASET.save_cached_sample(cache_path, sample)
    return idx


# ===========================================================================
# FACE ALIGNMENT UTILITY
# ===========================================================================

# Reference 5 facial landmarks for a canonical 112x112 aligned face
# (ArcFace standard, used by InsightFace, AltFreezing, SBI, etc.)
REFERENCE_LANDMARKS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


REFERENCE_LANDMARKS_224 = REFERENCE_LANDMARKS_112 * (224 / 112)


def align_face_5pt(img: np.ndarray, landmarks_5pt: np.ndarray, output_size: int = 224) -> np.ndarray:
    """
    Align a face image using 5 facial landmarks.
    Uses similarity transform (rotation + scale + translation only).
    img:           (H, W, 3) BGR or RGB numpy array
    landmarks_5pt: (5, 2) float32 — predicted [leye, reye, nose, lmouth, rmouth]
    output_size:   output image size (default 224)
    Returns: (output_size, output_size, 3) aligned face
    """
    ref = REFERENCE_LANDMARKS_224 * (output_size / 224)
    src = landmarks_5pt.astype(np.float32)

    tform = cv2.estimateAffinePartial2D(src, ref, method=cv2.LMEDS)[0]
    if tform is None:
        h, w = img.shape[:2]
        side  = min(h, w)
        top   = (h - side) // 2
        left  = (w - side) // 2
        cropped = img[top:top+side, left:left+side]
        return cv2.resize(cropped, (output_size, output_size))

    aligned = cv2.warpAffine(img, tform, (output_size, output_size),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return aligned


class FFPPDatasetV2(Dataset):
    """
        use_alignment:     bool  — align face using 5-pt landmarks (default True)
        training_mode:     bool  — enables augmentation in __getitem__
        label_smoothing:   float — smoothing factor for labels (default 0.05)
        use_sbi:           bool  — apply Self-Blended Images augmentation
        temporal_strategy: str  — 'motion' | 'uniform' | 'blend' (default 'blend')
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(
        self,
        video_paths,
        labels,
        mask_paths,
        domains,
        num_frames: int = 32,
        robust_mode=None,
        use_retinaface: bool = True,
        cache_dir=None,
        cache_version: str = "v2",        
        use_alignment: bool = True,        # face alignment
        training_mode: bool = False,       # enables augmentation at getitem
        label_smoothing: float = 0.05,     # smooth labels slightly
        use_sbi: bool = True,              # Self-Blended Images augmentation
        temporal_strategy: str = "blend",  # frame selection strategy
    ):
        self.video_paths       = video_paths
        self.labels            = labels
        self.mask_paths        = mask_paths
        self.domains           = domains
        self.num_frames        = num_frames
        self.robust_mode       = robust_mode
        self.use_retinaface    = use_retinaface
        self.cache_dir         = Path(cache_dir) if cache_dir is not None else None
        self.cache_version     = cache_version
        self.use_alignment     = use_alignment
        self.training_mode     = training_mode
        self.label_smoothing   = label_smoothing
        self.use_sbi           = use_sbi
        self.temporal_strategy = temporal_strategy
        self.gpu_id            = 0

        self.device      = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.face_detector = None

        self._video_aug = None
        self._sbi_aug   = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    # -----------------------------------------------------------------------
    # AUGMENTATION LAZY INIT
    # -----------------------------------------------------------------------
    def _get_augmenters(self):
        if self._video_aug is None and _AUG_AVAILABLE:
            self._video_aug = VideoAugmentation(training=True)
        if self._sbi_aug is None and _AUG_AVAILABLE and self.use_sbi:
            self._sbi_aug = SBIAugmentation(prob=0.25)
        return self._video_aug, self._sbi_aug

    # -----------------------------------------------------------------------
    # FACE DETECTOR
    # -----------------------------------------------------------------------
    def init_face_detector(self):
        if self.face_detector is not None:
            return
        if self.use_retinaface:
            providers, ctx_id = self.get_insightface_runtime()
            self.face_detector = FaceAnalysis(name="buffalo_l", providers=providers)
            self.face_detector.prepare(ctx_id=self.gpu_id, det_size=(640, 640))

    def get_insightface_runtime(self):
        if not torch.cuda.is_available():
            return ["CPUExecutionProvider"], -1
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        try:
            providers = ort.get_available_providers()
        except Exception:
            providers = []
        if "CUDAExecutionProvider" in providers:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
        return ["CPUExecutionProvider"], -1

    # -----------------------------------------------------------------------
    # DATASET 
    # -----------------------------------------------------------------------
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        cache_path = self.get_cache_path(idx)
        if cache_path is not None and cache_path.exists():
            sample = torch.load(cache_path, map_location="cpu")
        else:
            sample = self.build_sample(idx)
            if cache_path is not None:
                self.save_cached_sample(cache_path, sample)

        if self.training_mode and _AUG_AVAILABLE:
            video_aug, sbi_aug = self._get_augmenters()

            if sbi_aug is not None and sample["label"].item() == 0:
                sample = sbi_aug(sample)

            if video_aug is not None:
                from data.augmentation import augment_sample
                sample = augment_sample(sample, video_aug)

        if self.training_mode and self.label_smoothing > 0:
            lbl = sample["label"].item()
            eps = self.label_smoothing
            smooth = lbl * (1.0 - eps) + (1.0 - lbl) * eps
            sample = dict(sample)
            sample["label"] = torch.tensor(smooth, dtype=torch.float32)

        return sample

    # -----------------------------------------------------------------------
    # CACHE
    # -----------------------------------------------------------------------
    def cache_signature(self, idx):
        parts = [
            self.cache_version,
            self.video_paths[idx],
            str(self.mask_paths[idx]),
            str(self.labels[idx]),
            str(self.domains[idx]),
            str(self.num_frames),
            str(self.robust_mode),
            str(self.use_retinaface),
            str(self.use_alignment),      
            str(self.temporal_strategy), 
        ]
        return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()

    def get_cache_path(self, idx):
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.cache_signature(idx)}.pt"

    def save_cached_sample(self, cache_path, sample):
        temp_path = cache_path.with_suffix(".tmp")
        cpu_sample = {
            "frames":      sample["frames"].cpu(),
            "mask_frames": sample["mask_frames"].cpu(),
            "label":       sample["label"].cpu(),
            "has_mask":    bool(sample["has_mask"]),
            "domain":      sample["domain"],
        }
        torch.save(cpu_sample, temp_path)
        os.replace(temp_path, cache_path)

    def get_cache_stats(self):
        if self.cache_dir is None:
            return {"total": len(self), "cached": 0, "missing": len(self), "cache_dir": None}
        cached = sum(1 for i in range(len(self)) if self.get_cache_path(i).exists())
        return {
            "total":     len(self),
            "cached":    cached,
            "missing":   len(self) - cached,
            "cache_dir": str(self.cache_dir),
        }

    def export_init_kwargs(self):
        return {
            "video_paths":       self.video_paths,
            "labels":            self.labels,
            "mask_paths":        self.mask_paths,
            "domains":           self.domains,
            "num_frames":        self.num_frames,
            "robust_mode":       self.robust_mode,
            "use_retinaface":    self.use_retinaface,
            "cache_dir":         str(self.cache_dir) if self.cache_dir is not None else None,
            "cache_version":     self.cache_version,
            "use_alignment":     self.use_alignment,
            "training_mode":     False,   
            "label_smoothing":   0.0,
            "use_sbi":           False,
            "temporal_strategy": self.temporal_strategy,
        }

    def build_cache(self, overwrite=False, indices=None, num_workers=0):
        self.init_face_detector()
        if self.cache_dir is None:
            raise ValueError("Cache directory not configured.")

        before  = self.get_cache_stats()
        indices = indices if indices is not None else list(range(len(self)))

        pending = [i for i in indices if not self.get_cache_path(i).exists() or overwrite]
        progress = tqdm(total=len(indices), desc="Building Cache V2", dynamic_ncols=True)
        progress.update(len(indices) - len(pending))

        built = 0
        skipped = len(indices) - len(pending)
        worker_count = max(int(num_workers or 0), 1)

        if worker_count <= 1:
            for idx in pending:
                sample = self.build_sample(idx)
                self.save_cached_sample(self.get_cache_path(idx), sample)
                built += 1
                progress.update(1)
                if built % 200 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            dataset_kwargs = self.export_init_kwargs()
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(_build_cache_index_worker, dataset_kwargs, i) for i in pending]
                for f in as_completed(futures):
                    f.result()
                    built += 1
                    progress.update(1)

        progress.close()
        return self.get_cache_stats()

    # -----------------------------------------------------------------------
    # BUILD SAMPLE
    # -----------------------------------------------------------------------
    def build_sample(self, idx):
        video_path = self.video_paths[idx]
        label      = self.labels[idx]
        domain     = self.domains[idx]
        mask_path  = self.mask_paths[idx]

        frames, frame_indices = self.load_video_frames(video_path)

        raw_masks = self.load_mask_frames(mask_path, frame_indices) if mask_path is not None \
                    else [None] * len(frames)

        batch_results = self.extract_face_batch_aligned(frames)

        faces, aligned_masks = [], []
        for i, result in enumerate(batch_results):
            if result is None:
                continue
            face_tensor, bbox = result
            faces.append(self.rgb_ycbcr(face_tensor))

            mask = raw_masks[i]
            if mask is not None:
                aligned_masks.append(self.crop_mask(mask, bbox))
            elif mask_path is not None:
                aligned_masks.append(torch.zeros((224, 224), dtype=torch.float32))

        if len(faces) == 0:
            faces = [self.rgb_ycbcr(self.transform(Image.fromarray(f))) for f in frames]
            if mask_path is not None:
                aligned_masks = [
                    self.prepare_full_mask(m) if m is not None
                    else torch.zeros((224, 224), dtype=torch.float32)
                    for m in raw_masks
                ]

        if len(faces) < self.num_frames:
            faces         += [faces[-1]] * (self.num_frames - len(faces))
            aligned_masks += [aligned_masks[-1]] * (self.num_frames - len(aligned_masks)) \
                             if aligned_masks else []
        else:
            faces         = faces[:self.num_frames]
            aligned_masks = aligned_masks[:self.num_frames]

        faces_tensor = torch.stack(faces)

        if mask_path is not None and len(aligned_masks) > 0:
            mask_frames = torch.stack(aligned_masks)
            has_mask    = True
        else:
            mask_frames = torch.zeros((self.num_frames, 224, 224))
            has_mask    = False

        return {
            "frames":      faces_tensor,
            "mask_frames": mask_frames,
            "label":       torch.tensor(float(label), dtype=torch.float32),
            "has_mask":    has_mask,
            "domain":      domain,
        }

    def load_video_frames(self, path):
        try:
            vr          = VideoReader(path, ctx=cpu(0))
            total       = len(vr)
            if total <= 0:
                raise ValueError(f"Empty video: {path}")

            sample_count = min(total, self.num_frames * 4)
            indices      = np.linspace(0, total - 1, sample_count).astype(int)
            frames_array = vr.get_batch(indices).asnumpy()
            frames       = list(frames_array)

            if len(frames) == 0:
                raise ValueError(f"No frames decoded: {path}")

            selected = self._select_frames(frames, indices)
            return [frames[i] for i in selected], [int(indices[i]) for i in selected]

        except Exception as e:
            return self._load_opencv_fallback(path)

    def _select_frames(self, frames, all_indices):
        """
        Blended selection strategy.
        Pure motion can cluster all frames in one scene cut.
        Blend = 0.6 * motion_rank + 0.4 * uniform_pressure.

        Temporal stride selection (uniform + motion-weighted blend)
        Instead of pure motion-based selection (which can miss important
        static frames),  blends motion scores with uniform sampling pressure.
        This prevents clustering all selected frames in one scene transition.
        """
        n = len(frames)
        if n <= self.num_frames:
            return list(range(n))

        motion_scores = self._motion_scores(frames)

        if self.temporal_strategy == "motion":
            selected = np.argsort(motion_scores)[-self.num_frames:]
            return sorted(selected)

        elif self.temporal_strategy == "uniform":
            return sorted(np.linspace(0, n - 1, self.num_frames).astype(int).tolist())

        else:  # 'blend' — default
            motion_rank  = np.argsort(np.argsort(motion_scores)).astype(float) / max(n - 1, 1)
            uniform_rank = np.linspace(0, 1, n)
            combined     = 0.6 * motion_rank + 0.4 * uniform_rank
            selected     = np.argsort(combined)[-self.num_frames:]
            return sorted(selected)

    def _motion_scores(self, frames):
        scores = [0.0]
        for i in range(1, len(frames)):
            prev = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(frames[i],   cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr,
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            scores.append(float(np.mean(mag)))
        return scores

    def _load_opencv_fallback(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_count = min(max(total, 1), self.num_frames * 4)
        indices = np.linspace(0, max(total - 1, 0), sample_count).astype(int)
        frames, valid = [], []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                valid.append(fi)
        cap.release()
        if not frames:
            raise ValueError(f"No frames: {path}")
        selected = self._select_frames(frames, valid)
        return [frames[i] for i in selected], [int(valid[i]) for i in selected]

    # -----------------------------------------------------------------------
    # This removes pose variation and lets the network focus on manipulation artifacts instead of pose differences. Alignment makes blending boundaries consistent across frames.
    # -----------------------------------------------------------------------
    def extract_face_batch_aligned(self, frames):
        """
        Extract and ALIGN faces using 5-point landmarks.
        Returns list of (tensor, bbox) or None per frame.
        """
        results = []
        for frame in frames:
            if frame is None:
                results.append(None)
                continue

            try:
                faces = self.face_detector.get(frame)
            except Exception:
                results.append(None)
                continue

            if not faces:
                results.append(None)
                continue

            face = faces[0]
            x1, y1, x2, y2 = map(int, face.bbox)
            x1, y1, x2, y2 = self.expand_and_clip_bbox(
                x1, y1, x2, y2, frame.shape[1], frame.shape[0]
            )

            if self.use_alignment and hasattr(face, 'kps') and face.kps is not None:
                try:
                    aligned = align_face_5pt(frame, face.kps, output_size=224)
                    face_tensor = torch.from_numpy(aligned).permute(2, 0, 1).float() / 255.0
                except Exception:
                    # fallback to bbox crop
                    face_img    = Image.fromarray(frame[y1:y2, x1:x2])
                    face_tensor = self.transform(face_img)
            else:
                face_img    = Image.fromarray(frame[y1:y2, x1:x2])
                face_tensor = self.transform(face_img)

            results.append((face_tensor, (x1, y1, x2, y2)))

        return results

    def load_mask_frames(self, path, indices):
        try:
            vr          = VideoReader(path, ctx=cpu(0))
            total       = len(vr)
            safe        = [min(int(i), total - 1) for i in indices]
            frames_arr  = vr.get_batch(safe).asnumpy()
            return [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_arr]
        except Exception:
            return [None] * len(indices)

    def expand_and_clip_bbox(self, x1, y1, x2, y2, width, height):
        w = x2 - x1
        h = y2 - y1
        x1 = max(int(x1 - 0.15 * w), 0)
        y1 = max(int(y1 - 0.15 * h), 0)
        x2 = min(int(x2 + 0.15 * w), width)
        y2 = min(int(y2 + 0.15 * h), height)
        return x1, y1, x2, y2

    def crop_mask(self, mask, bbox):
        x1, y1, x2, y2 = bbox
        mask = mask[y1:y2, x1:x2]
        if mask.size == 0:
            return torch.zeros((224, 224), dtype=torch.float32)
        mask = cv2.resize(mask, (224, 224)) / 255.0
        return torch.tensor(mask, dtype=torch.float32)

    def prepare_full_mask(self, mask):
        mask = cv2.resize(mask, (224, 224)) / 255.0
        return torch.tensor(mask, dtype=torch.float32)

    def rgb_ycbcr(self, tensor_img: torch.Tensor) -> torch.Tensor:
        """
        Convert 3-channel [0,1] tensor to 6-channel [-1,1] RGB+YCbCr.
        FIX-4: output is explicitly clamped to avoid BN NaN from extreme values.
        """
        r, g, b = tensor_img[0], tensor_img[1], tensor_img[2]
        y  =  0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
        ycbcr    = torch.stack([y, cb, cr], dim=0)
        combined = torch.cat([tensor_img, ycbcr], dim=0)
        normalized = (combined - 0.5) / 0.5
        return torch.clamp(normalized, -1.0, 1.0)   # FIX-4

def get_sampler_weights(dataset) -> list:
    """
    Read labels directly from the dataset's label list — O(N) no disk access.
    Works with both FFPPDatasetV2 directly and torch.utils.data.Subset.
    """
    if hasattr(dataset, "labels"):
        labels = [int(l) for l in dataset.labels]
    elif hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        # Subset
        base   = dataset.dataset
        labels = [int(base.labels[i]) for i in dataset.indices]
    else:
        # Slow fallback
        labels = [int(dataset[i]["label"].item()) for i in range(len(dataset))]

    n_real = labels.count(0)
    n_fake = labels.count(1)
    w_real = 1.0 / (n_real + 1e-6)
    w_fake = 1.0 / (n_fake + 1e-6)
    return [w_real if l == 0 else w_fake for l in labels]
