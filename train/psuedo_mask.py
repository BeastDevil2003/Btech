import torch
import torch.nn.functional as F


def _temporal_ema(cam, decay):
    if cam.shape[1] <= 1:
        return cam
    smoothed = torch.empty_like(cam)
    smoothed[:, 0] = cam[:, 0]
    for t in range(1, cam.shape[1]):
        smoothed[:, t] = decay * smoothed[:, t - 1] + (1 - decay) * cam[:, t]
    return smoothed


def compute_gradcam_stable(model, frames, decay=0.9):
    frames_cam = frames.clone().detach().requires_grad_(True)
    out        = model(frames_cam)

    score      = torch.sigmoid(out["pred_cls"]).mean()
    grads      = torch.autograd.grad(score, frames_cam, retain_graph=False)[0]
    cam        = grads.abs().mean(dim=2, keepdim=True)
    cam        = F.relu(cam)
    flat       = cam.view(cam.shape[0], -1)
    cam_min    = flat.min(dim=1)[0].view(-1, 1, 1, 1, 1)
    cam_max    = flat.max(dim=1)[0].view(-1, 1, 1, 1, 1)
    cam        = (cam - cam_min) / (cam_max - cam_min + 1e-6)
    return _temporal_ema(cam.detach(), decay)


def frequency_map(Ff):
    fmap = torch.mean(torch.abs(Ff), dim=1, keepdim=True)
    return F.interpolate(fmap, size=(224, 224), mode="bilinear", align_corners=False)


def noise_map(Fn):
    nmap = torch.mean(torch.abs(Fn), dim=1, keepdim=True)
    return F.interpolate(nmap, size=(224, 224), mode="bilinear", align_corners=False)


def _restore_temporal_layout(feature_map, frames):
    if feature_map.dim() == 5:
        return feature_map
    B, N = frames.shape[:2]
    return feature_map.view(B, N, feature_map.shape[1],
                            feature_map.shape[2], feature_map.shape[3])


def generate_pseudo_mask(model, frames, Fs, Ff, Fn,
                         alpha=0.5, beta=0.3, gamma=0.2,
                         use_gradcam=True):
    """
    use_gradcam=False  → uses Ff + Fn only, no 2nd forward pass → VRAM safe.
    Always pass use_gradcam=False during training on a 4 GB GPU.
    Never call with use_gradcam=True inside torch.no_grad().
    """
    fmap = _restore_temporal_layout(frequency_map(Ff), frames)
    nmap = _restore_temporal_layout(noise_map(Fn),     frames)

    if use_gradcam:
        cam    = compute_gradcam_stable(model, frames)
        pseudo = alpha * cam + beta * fmap + gamma * nmap
    else:
        b      = beta  / (beta + gamma)
        g      = gamma / (beta + gamma)
        pseudo = b * fmap + g * nmap

    flat       = pseudo.view(pseudo.shape[0], -1)
    pseudo_min = flat.min(dim=1)[0].view(-1, 1, 1, 1, 1)
    pseudo_max = flat.max(dim=1)[0].view(-1, 1, 1, 1, 1)
    return ((pseudo - pseudo_min) / (pseudo_max - pseudo_min + 1e-6)).detach()
