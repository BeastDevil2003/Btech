"""
augmentation.py  — Video-consistent augmentation for deepfake detection training
=================================================================================
Key design principle:
    ALL augmentations make ONE random decision and apply it IDENTICALLY to
    every frame in the video sequence. Applying different transforms per frame
    would create artificial temporal inconsistencies that bias the temporal model.

Techniques included:
    1. JPEG compression simulation — critical for FF++ generalization
       (FF++ C23 uses heavy compression; models must handle varied quality)
    2. Random Gaussian noise — robustness augmentation
    3. Color jitter — subtle face-level variation
    4. Horizontal flip (video-consistent)
    5. Random erasing — occlusion robustness
    6. Frequency-domain noise — exposes models to frequency artifacts during training
    
Usage:
    # In your training loop or dataset __getitem__:
    augmenter = VideoAugmentation(training=True)
    frames_tensor = augmenter(frames_tensor)  # (N, C, H, W)
    
    # frames_tensor should be in [-1, 1] range (your rgb_ycbcr normalization)
    
Research basis:
    SBI (CVPR 2022): self-blended augmentation for generalization
    LipForensics (CVPR 2021): augmentation for temporal deepfake detection
    LSDA (ICCV 2023): large-scale augmentation strategy
"""

import io
import random
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class VideoAugmentation:
    """
    Video-level augmentation: ONE random decision per video, applied to ALL frames.
    
    Input/Output tensor shape: (N, C, H, W) where
        N = number of frames (e.g. 16)
        C = channels (6 for RGB+YCbCr, or 3)
        H, W = 224, 224
        Value range: [-1, 1]  (your rgb_ycbcr normalization)
    """

    def __init__(
        self,
        training: bool = True,
        # Augmentation probabilities
        flip_prob: float = 0.5,
        jpeg_prob: float = 0.4,
        noise_prob: float = 0.3,
        color_jitter_prob: float = 0.3,
        erase_prob: float = 0.1,
        freq_noise_prob: float = 0.2,
        # Augmentation parameters
        jpeg_quality_range: Tuple[int, int] = (55, 95),
        noise_std_range: Tuple[float, float] = (0.005, 0.03),
        brightness_range: Tuple[float, float] = (-0.1, 0.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        saturation_range: Tuple[float, float] = (0.95, 1.05),
        erase_scale_range: Tuple[float, float] = (0.02, 0.12),
        # Channel mode: 'rgb' for 3ch, 'rgb_ycbcr' for 6ch
        channel_mode: str = "rgb_ycbcr",
    ):
        self.training         = training
        self.flip_prob        = flip_prob
        self.jpeg_prob        = jpeg_prob
        self.noise_prob       = noise_prob
        self.color_jitter_prob = color_jitter_prob
        self.erase_prob       = erase_prob
        self.freq_noise_prob  = freq_noise_prob

        self.jpeg_quality_range  = jpeg_quality_range
        self.noise_std_range     = noise_std_range
        self.brightness_range    = brightness_range
        self.contrast_range      = contrast_range
        self.saturation_range    = saturation_range
        self.erase_scale_range   = erase_scale_range
        self.channel_mode        = channel_mode

    # -----------------------------------------------------------------------
    # Individual transforms
    # -----------------------------------------------------------------------

    def horizontal_flip(self, x: torch.Tensor) -> torch.Tensor:
        """Flip all frames horizontally. x: (N, C, H, W)"""
        return torch.flip(x, dims=[-1])

    def jpeg_simulate(self, x: torch.Tensor, quality: int) -> torch.Tensor:
        """
        Simulate JPEG compression artifacts.
        
        Critical for FF++ generalization:
            FF++ C23 = quality 23 compression during dataset creation
            Training without JPEG aug → model learns non-compressed artifacts only
            → Fails on social media / real-world videos which are re-compressed
        
        x: (N, C, H, W) in [-1, 1], operates only on first 3 (RGB) channels
        """
        if not PIL_AVAILABLE:
            return x  # skip if PIL not available

        N, C, H, W = x.shape
        # Only apply to RGB channels (first 3) to avoid corrupting YCbCr
        rgb_channels = min(3, C)

        # Convert from [-1, 1] → [0, 255]
        x_out = x.clone()
        for n in range(N):
            try:
                frame_rgb = x[n, :rgb_channels]
                frame_uint8 = ((frame_rgb + 1.0) / 2.0 * 255.0).clamp(0, 255)
                frame_uint8 = frame_uint8.byte().permute(1, 2, 0).cpu().numpy()

                pil_img = Image.fromarray(frame_uint8)
                buffer  = io.BytesIO()
                pil_img.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                pil_compressed = Image.open(buffer).convert("RGB")

                compressed = torch.tensor(
                    np.array(pil_compressed), dtype=x.dtype, device=x.device
                ).permute(2, 0, 1)   # (3, H, W)

                # Back to [-1, 1]
                x_out[n, :rgb_channels] = (compressed / 255.0) * 2.0 - 1.0
            except Exception:
                pass  # If JPEG fails for any frame, skip it

        return x_out

    def gaussian_noise(self, x: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise, same std across frames, different noise realization."""
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, -1.0, 1.0)

    def color_jitter(
        self,
        x: torch.Tensor,
        brightness: float,
        contrast: float,
        saturation: float,
    ) -> torch.Tensor:
        """
        Subtle face-level color augmentation.
        Uses same parameters for all frames to maintain temporal consistency.
        Only applied to RGB channels (first 3).
        """
        x_out = x.clone()
        rgb_channels = min(3, x.shape[1])

        for n in range(x.shape[0]):
            frame = x_out[n, :rgb_channels]  # (3, H, W) in [-1, 1]

            # Brightness: add constant
            frame = frame + brightness

            # Contrast: scale around mean
            mean = frame.mean(dim=[1, 2], keepdim=True)
            frame = mean + contrast * (frame - mean)

            # Saturation: interpolate between frame and grayscale
            # (Only meaningful for RGB channels)
            if rgb_channels == 3:
                gray = 0.299 * frame[0] + 0.587 * frame[1] + 0.114 * frame[2]
                gray = gray.unsqueeze(0).expand_as(frame)
                frame = saturation * frame + (1.0 - saturation) * gray

            x_out[n, :rgb_channels] = torch.clamp(frame, -1.0, 1.0)

        return x_out

    def random_erase(
        self,
        x: torch.Tensor,
        scale: float,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """
        Erase a random rectangular region — same region for all frames.
        Teaches the model to detect fakes from partial face views.
        """
        N, C, H, W = x.shape
        area    = H * W * scale
        aspect  = random.uniform(0.5, 2.0)
        h_erase = int((area / aspect) ** 0.5)
        w_erase = int((area * aspect) ** 0.5)

        h_erase = min(h_erase, H)
        w_erase = min(w_erase, W)

        top  = random.randint(0, H - h_erase)
        left = random.randint(0, W - w_erase)

        x_out = x.clone()
        x_out[:, :, top:top + h_erase, left:left + w_erase] = fill_value
        return x_out

    def frequency_domain_noise(self, x: torch.Tensor, strength: float = 0.05) -> torch.Tensor:
        """
        Add noise in the frequency domain — simulates GAN artifacts.
        
        Real GAN images produce specific frequency patterns (spectral artifacts).
        By training with frequency noise, the model learns to generalize across
        different GAN architectures' artifact signatures.
        
        Works on float32 internally; casts back to original dtype.
        """
        orig_dtype = x.dtype
        x_f32 = x.float()  # Always float32 for FFT

        N, C, H, W = x_f32.shape

        # FFT per frame
        fft = torch.fft.rfft2(x_f32, norm='ortho')

        # Add small random perturbation to high-frequency components
        # (low-frequency is left clean to preserve face structure)
        noise_mag = torch.randn_like(fft.real) * strength
        noise_phase = torch.randn_like(fft.real) * strength * 0.5

        # Apply only to high-frequency half of spectrum
        H_mid = fft.shape[-2] // 2
        W_mid = fft.shape[-1] // 2
        fft.real[:, :, H_mid:, W_mid:] += noise_mag[:, :, H_mid:, W_mid:]
        fft.imag[:, :, H_mid:, W_mid:] += noise_phase[:, :, H_mid:, W_mid:]

        # Inverse FFT
        x_perturbed = torch.fft.irfft2(fft, s=(H, W), norm='ortho')
        x_perturbed = torch.clamp(x_perturbed, -1.0, 1.0)

        return x_perturbed.to(orig_dtype)

    # -----------------------------------------------------------------------
    # Main call
    # -----------------------------------------------------------------------

    def __call__(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply video-consistent augmentation.
        
        frames_tensor: (N, C, H, W), values in [-1, 1]
        returns: augmented tensor, same shape and dtype
        """
        if not self.training:
            return frames_tensor

        x = frames_tensor

        # --- 1. Horizontal flip (single random decision for all frames) ---
        if random.random() < self.flip_prob:
            x = self.horizontal_flip(x)

        # --- 2. JPEG compression simulation ---
        if random.random() < self.jpeg_prob:
            quality = random.randint(*self.jpeg_quality_range)
            x = self.jpeg_simulate(x, quality)

        # --- 3. Color jitter ---
        if random.random() < self.color_jitter_prob:
            brightness = random.uniform(*self.brightness_range)
            contrast   = random.uniform(*self.contrast_range)
            saturation = random.uniform(*self.saturation_range)
            x = self.color_jitter(x, brightness, contrast, saturation)

        # --- 4. Gaussian noise ---
        if random.random() < self.noise_prob:
            std = random.uniform(*self.noise_std_range)
            x = self.gaussian_noise(x, std)

        # --- 5. Random erasing ---
        if random.random() < self.erase_prob:
            scale = random.uniform(*self.erase_scale_range)
            x = self.random_erase(x, scale)

        # --- 6. Frequency domain noise ---
        if random.random() < self.freq_noise_prob:
            strength = random.uniform(0.01, 0.05)
            x = self.frequency_domain_noise(x, strength)

        return x


# -----------------------------------------------------------------------
# Integration helper: wrap your dataset __getitem__ with this
# -----------------------------------------------------------------------
def augment_sample(sample: dict, augmenter: VideoAugmentation) -> dict:
    """
    Applies augmentation to a cached dataset sample dict.
    
    Expected keys: "frames" (N, C, H, W), "mask_frames" (N, H, W), etc.
    The mask is NOT augmented (only flipped if frames are flipped — handle separately).
    
    Usage:
        augmenter = VideoAugmentation(training=True)
        
        # In your training dataset's __getitem__:
        sample = self.load_from_cache(idx)
        sample = augment_sample(sample, augmenter)
    """
    augmented = dict(sample)
    frames = sample["frames"]  # (N, C, H, W)

    if augmenter.training:
        # Decide flip BEFORE augmentation
        do_flip = random.random() < augmenter.flip_prob

        if do_flip:
            frames = torch.flip(frames, dims=[-1])
            # Also flip masks to maintain spatial correspondence
            if "mask_frames" in sample and sample["mask_frames"] is not None:
                augmented["mask_frames"] = torch.flip(sample["mask_frames"], dims=[-1])

        # Apply remaining augmentations (flip was already applied or skipped)
        original_flip_prob = augmenter.flip_prob
        augmenter.flip_prob = 0.0  # disable flip in main call since we handled it
        frames = augmenter(frames)
        augmenter.flip_prob = original_flip_prob

    augmented["frames"] = frames
    return augmented


# -----------------------------------------------------------------------
# Self-Blended Images (SBI) augmentation — highest-impact for generalization
# Reference: "Detecting Deepfakes with Self-Blended Images" (CVPR 2022)
# -----------------------------------------------------------------------
class SBIAugmentation:
    """
    Creates synthetic fake training samples by blending regions of REAL faces.
    
    Algorithm:
    1. Take a real face frame
    2. Warp/transform a copy of it slightly (affine, elastic deformation)
    3. Create a binary or smooth blending mask
    4. Blend: fake_frame = mask * warped + (1-mask) * original
    5. Label the result as FAKE (1) with the blending mask as ground truth
    
    This teaches the model to detect blending boundaries without needing
    a diverse collection of fake generation methods.
    
    Simple version provided here — for production use, see the original paper.
    """

    def __init__(
        self,
        prob: float = 0.3,
        alpha_range: Tuple[float, float] = (0.3, 0.7),
    ):
        self.prob        = prob
        self.alpha_range = alpha_range

    def _simple_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple affine warp — approximate face landmark displacement.
        x: (C, H, W) single frame in [-1, 1]
        """
        C, H, W = x.shape
        # Small random affine transform
        angle = random.uniform(-5, 5) * (torch.pi / 180)
        scale = random.uniform(0.95, 1.05)
        tx    = random.uniform(-5, 5) / W
        ty    = random.uniform(-5, 5) / H

        cos_a = torch.tensor([[scale * torch.cos(angle), -scale * torch.sin(angle), tx],
                              [scale * torch.sin(angle),  scale * torch.cos(angle), ty]])

        grid = F.affine_grid(cos_a.unsqueeze(0), (1, C, H, W), align_corners=False)
        return F.grid_sample(x.unsqueeze(0), grid, align_corners=False, padding_mode='border').squeeze(0)

    def _gaussian_mask(self, H: int, W: int, device) -> torch.Tensor:
        """
        Generate a smooth elliptical blending mask centered in the face region.
        """
        cy, cx = H // 2 + random.randint(-H//8, H//8), W // 2 + random.randint(-W//8, W//8)
        ry     = H // 4 + random.randint(-H//8, H//8)
        rx     = W // 4 + random.randint(-W//8, W//8)

        yy, xx = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device), indexing='ij')
        mask = ((yy - cy) ** 2 / (ry ** 2 + 1e-6) + (xx - cx) ** 2 / (rx ** 2 + 1e-6))
        mask = torch.exp(-mask)
        # Soft threshold
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        return mask.unsqueeze(0)  # (1, H, W)

    def __call__(self, sample: dict) -> dict:
        """
        With probability self.prob, convert a REAL sample to a SBI fake sample.
        
        Only applies to REAL samples (label=0).
        Returns modified sample with label=1 and pseudo blending mask.
        """
        if not torch.is_tensor(sample["label"]):
            label = float(sample["label"])
        else:
            label = sample["label"].item()

        if label != 0 or random.random() > self.prob:
            return sample

        frames = sample["frames"]   # (N, C, H, W)
        N, C, H, W = frames.shape

        # Blend all frames with the SAME mask but DIFFERENT warp realizations
        alpha     = random.uniform(*self.alpha_range)
        blend_mask = self._gaussian_mask(H, W, frames.device)  # (1, H, W)

        blended_frames = []
        for n in range(N):
            frame       = frames[n]         # (C, H, W)
            warped      = self._simple_warp(frame)
            blended     = alpha * blend_mask * warped + (1 - alpha * blend_mask) * frame
            blended_frames.append(blended.clamp(-1.0, 1.0))

        new_sample = dict(sample)
        new_sample["frames"]      = torch.stack(blended_frames)
        new_sample["label"]       = torch.tensor(1.0, dtype=sample["label"].dtype if torch.is_tensor(sample["label"]) else torch.float32)
        new_sample["mask_frames"] = blend_mask.squeeze(0).expand(N, H, W)
        new_sample["has_mask"]    = True

        return new_sample
