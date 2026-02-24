from io import BytesIO
import base64
import os
from typing import Optional
from PIL import Image as PILImage


from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModel


app = FastAPI()
SERVER_API_KEY = os.getenv("MODEL_SERVER_API_KEY", "agm_srv_f6e9b635ff9025b982f7f77e240e8510a48ba1fad073ad7f")


def _check_api_key(authorization: Optional[str]) -> None:
    if not SERVER_API_KEY:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="missing Authorization header")

    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token != SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = AutoModel.from_pretrained(
    "jinaai/jina-clip-v2",
    trust_remote_code=True,
).to(device)
clip_model.eval()


class TextsReq(BaseModel):
    texts: list[str]


class ImagesReq(BaseModel):
    images: Optional[list[str]] = None
    images_base64: Optional[list[str]] = None


@app.post("/clip/text")
def clip_text(req: TextsReq, authorization: Optional[str] = Header(default=None)):
    _check_api_key(authorization)
    embs = clip_model.encode_text(req.texts)
    return {"dense": [e.tolist() if hasattr(e, "tolist") else e for e in embs]}


@app.post("/clip/image")
def clip_image(req: ImagesReq, authorization: Optional[str] = Header(default=None)):
    _check_api_key(authorization)
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
