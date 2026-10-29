"""Shared inference components for the educational DR prototype."""
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parent
LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CHECKPOINT = ROOT / "models" / "best.pt"
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]


IMAGE_SIZE = 256


def preprocess_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """Apply the training notebook's deterministic evaluation resize (no augmentation)."""
    pil = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    if min(pil.size) < 100:
        raise ValueError("Image must have both sides at least 100 pixels.")
    return np.asarray(pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR))


def preprocess(image: Image.Image | np.ndarray) -> torch.Tensor:
    rgb = preprocess_array(image)
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.)
    return (x - MEAN) / STD


def quality_check(image: Image.Image) -> tuple[bool, str]:
    """Reject only images too small to process; report visual heuristics as advisory."""
    if not isinstance(image, Image.Image):
        return False, "The uploaded file is not a readable image."
    try:
        rgb = np.asarray(image.convert("RGB"))
    except (OSError, ValueError):
        return False, "The uploaded file is not a readable image."
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 100:
        return False, "Image is too small to process (minimum side: 100 pixels)."

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Fundus photos commonly have dark circular borders. Estimate focus on the
    # brighter image area, resized to a fixed scale so the score is less tied to
    # the source resolution. This remains an advisory heuristic: retinal images
    # can naturally have few sharp edges.
    threshold = max(5, int(np.percentile(gray, 90) * 0.08))
    mask = gray > threshold
    roi_mean = float(np.mean(gray[mask])) if np.any(mask) else float(gray.mean())
    ys, xs = np.where(mask)
    if len(xs) and len(ys):
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if x1 - x0 >= 50 and y1 - y0 >= 50:
            gray = gray[y0:y1, x0:x1]
    focus_image = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    if float(cv2.Laplacian(focus_image, cv2.CV_64F).var()) < 8:
        return True, "Image may be blurry or low in visible detail. You can still analyze it; check the result carefully."
    if not 10 <= roi_mean <= 245:
        return True, "Image brightness may limit reliability. You can still analyze it; check the result carefully."
    return True, "Passed basic image-quality heuristics; this does not verify that it is a retinal image."


def build_model(pretrained: bool = False):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    return model


def load_predictor():
    if not CHECKPOINT.exists():
        return None, f"Checkpoint not found at {CHECKPOINT}. Train a model before requesting predictions."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = build_model(False)
        saved = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        if isinstance(saved, dict) and saved.get("architecture", "efficientnet_b0") != "efficientnet_b0":
            raise ValueError(f"Unsupported checkpoint architecture: {saved['architecture']}")
        if isinstance(saved, dict) and saved.get("image_size", IMAGE_SIZE) != IMAGE_SIZE:
            raise ValueError(f"Checkpoint image size does not match app input size ({IMAGE_SIZE}).")
        state = saved.get("model_state_dict", saved) if isinstance(saved, dict) else saved
        model.load_state_dict(state)
        model.to(device).eval()
        return (model, device), None
    except Exception as exc:
        return None, f"Checkpoint could not be loaded: {exc}"


@torch.inference_mode()
def predict(predictor, image: Image.Image) -> dict:
    try:
        x = preprocess(image).unsqueeze(0).to(predictor[1])
        probs = torch.softmax(predictor[0](x), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return {"ok": True, "label": LABELS[idx], "confidence": float(probs[idx]),
                "probabilities": {LABELS[i]: float(v) for i, v in enumerate(probs)},
                "grade": idx, "referable_demo_group": idx >= 2, "input_tensor": x}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


def gradcam(predictor, image: Image.Image, class_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a grayscale heatmap and RGB overlay for EfficientNet's final feature block."""
    import cv2
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    model, device = predictor
    x = preprocess(image).unsqueeze(0).to(device)
    rgb = preprocess_array(image)
    cam = GradCAM(model=model, target_layers=[model.features[-1]])
    try:
        # Grad-CAM needs gradients even when called from inference-oriented UI code.
        with torch.enable_grad():
            heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(class_index))])[0]
        color = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        overlay = show_cam_on_image(rgb.astype(np.float32) / 255.0, heat, use_rgb=True)
        return heatmap, overlay
    finally:
        cam.activations_and_grads.release()
