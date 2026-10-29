import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from app_core import preprocess, load_predictor
def test_preprocess_shape():
    x=preprocess(Image.fromarray(np.full((160,180,3),120,dtype=np.uint8)))
    assert tuple(x.shape)==(3,256,256)
def test_missing_checkpoint_safe(tmp_path,monkeypatch):
    import app_core
    monkeypatch.setattr(app_core,"CHECKPOINT",tmp_path/"missing.pt")
    model,error=load_predictor()
    assert model is None and "Checkpoint not found" in error

def test_preprocessing_is_deterministic():
    image = Image.fromarray(np.random.default_rng(3).integers(0, 255, (180, 220, 3), dtype=np.uint8))
    assert np.array_equal(preprocess(image).numpy(), preprocess(image).numpy())

def _fundus_like_image():
    image = Image.new("RGB", (320, 320), (3, 3, 3))
    draw = ImageDraw.Draw(image)
    draw.ellipse((25, 25, 295, 295), fill=(100, 55, 45))
    draw.ellipse((120, 115, 150, 145), fill=(180, 100, 80))
    for end in [(40, 80), (270, 70), (45, 250), (275, 250), (160, 35), (160, 280)]:
        draw.line((135, 130, *end), fill=(210, 145, 125), width=4)
    return image

def test_quality_check_accepts_clear_fundus_with_dark_border():
    from app_core import quality_check
    accepted, message = quality_check(_fundus_like_image())
    assert accepted
    assert "may be blurry" not in message

def test_quality_check_warns_but_allows_blurry_image():
    from app_core import quality_check
    accepted, message = quality_check(_fundus_like_image().filter(ImageFilter.GaussianBlur(radius=12)))
    assert accepted
    assert "blurry" in message

def test_quality_check_rejects_unprocessable_image():
    from app_core import quality_check
    accepted, message = quality_check(Image.new("RGB", (80, 120), "black"))
    assert not accepted
    assert "too small" in message

def test_guidance_falls_back_without_retrieved_sources(monkeypatch):
    import rag_core
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    text, status = rag_core.guidance(2, [])
    assert "cannot provide individualized guidance" in text
    assert status == "No retrieved evidence"

def test_guidance_falls_back_without_groq_credentials(monkeypatch):
    import rag_core
    monkeypatch.setattr(rag_core, "ROOT", __import__("pathlib").Path("."))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    text, status = rag_core.guidance(1, [{"citation": "[1]", "source": "guide.pdf", "page": 2,
                                          "text": "A reviewed educational excerpt.", "score": .8}])
    assert "cannot provide individualized guidance" in text
    assert status == "GROQ_API_KEY is not configured"

def test_mongodb_is_optional_without_uri(monkeypatch):
    from metadata_store import save_record
    monkeypatch.setenv("MONGODB_URI", "")
    ok, message = save_record({"demo_id": "demo-001", "image_bytes": b"private image"})
    assert not ok
    assert "not configured" in message
