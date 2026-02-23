from io import BytesIO
import base64
from PIL import Image as PILImage


from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor


app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = AutoModel.from_pretrained(
    "jinaai/jina-clip-v2",
    trust_remote_code=True,
).to(device)
clip_model.eval()

# 强制使用本地缓存，不再联网
clip_model.tokenizer = AutoTokenizer.from_pretrained(
    "jinaai/jina-clip-v2",
    trust_remote_code=True,
    local_files_only=True,
)
clip_model.preprocess = AutoImageProcessor.from_pretrained(
    "jinaai/jina-clip-v2",
    trust_remote_code=True,
    local_files_only=True,
)


class TextsReq(BaseModel):
    texts: list[str]


class ImagesReq(BaseModel):
    images: list[str] | None = None
    images_base64: list[str] | None = None


@app.post("/clip/text")
def clip_text(req: TextsReq):
    embs = clip_model.encode_text(req.texts)
    return {"dense": [e.tolist() if hasattr(e, "tolist") else e for e in embs]}


@app.post("/clip/image")
def clip_image(req: ImagesReq):
    images = []
    if getattr(req, "images_base64", None):
        for b64 in req.images_base64:
            data = base64.b64decode(b64)
            img = PILImage.open(BytesIO(data))
            if getattr(img, "is_animated", False):
                img.seek(0)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            images.append(img)
    else:
        images = req.images  # 兼容原有路径/URL

    embs = clip_model.encode_image(images)
    return {"dense": [e.tolist() if hasattr(e, "tolist") else e for e in embs]}
