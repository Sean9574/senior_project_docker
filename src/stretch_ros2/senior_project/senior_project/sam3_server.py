#!/usr/bin/env python3
"""\
SAM3 Segmentation Server
Single prompt controls all segmentation requests.

This version is optimized for FAST video segmentation while still using SAM3
segmentation masks.

Main speed features:
- "combined" mask mode: return ONE merged mask instead of N instance masks.
- Optional mask downscale (e.g., 256px long side) to reduce payload and
  encode/decode cost.
- Top-K + min-area filtering to reduce noise and unnecessary work.

Usage:
    conda activate sam3
    python sam3_server.py
"""

import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

# ============== GLOBAL STATE ==============
model = None
processor = None
CURRENT_PROMPT = "object"
# ==========================================


class SegmentRequest(BaseModel):
    image_base64: str
    prompt: Optional[str] = None

    # Filter detections
    confidence_threshold: float = 0.30
    max_objects: int = 50

    # Mask output
    mask_mode: Literal["instances", "combined"] = "instances"
    mask_threshold: float = 0.50
    mask_size: int = 0  # 0 = keep model output resolution; else downscale long side to this
    min_mask_area_frac: float = 0.0  # e.g., 0.0005 = 0.05% of pixels

    # Kept for API compatibility (not implemented here)
    return_visualization: bool = False


class SegmentResponse(BaseModel):
    success: bool
    prompt: str
    num_objects: int
    masks_base64: list[str]
    boxes: list[list[float]]
    scores: list[float]
    inference_time_ms: float
    visualization_base64: Optional[str] = None
    error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor

    print("=" * 60)
    print("Loading SAM3 model...")
    print("=" * 60)

    # Safe inference speedups
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    start = time.time()
    model = build_sam3_image_model()
    if torch.cuda.is_available():
        model = model.cuda().eval()
    else:
        model = model.eval()

    processor = Sam3Processor(model)

    # Warmup to avoid first-request hiccup
    try:
        dummy = Image.new("RGB", (256, 256), (0, 0, 0))
        with torch.inference_mode():
            st = processor.set_image(dummy)
            _ = processor.set_text_prompt(state=st, prompt="object")
    except Exception:
        pass

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"✓ SAM3 loaded in {time.time() - start:.1f}s")
    print(f"✓ Device: {device_str}")
    print(f"✓ Default prompt: '{CURRENT_PROMPT}'")
    print("✓ Server ready at http://0.0.0.0:8100")
    print("=" * 60)

    yield

    del model, processor
    model = None
    processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="SAM3 Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def decode_image(base64_str: str) -> Image.Image:
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")


def _mask_to_u8(mask: np.ndarray, mask_threshold: float) -> np.ndarray:
    """Convert a SAM3 mask array to uint8 {0,255}."""
    m = mask
    while m.ndim > 2:
        m = m[0]

    if m.dtype == bool:
        out = m.astype(np.uint8) * 255
        return out

    m = m.astype(np.float32)

    # If it looks like probabilities, threshold directly; otherwise normalize then threshold.
    if 0.0 <= float(m.min()) and float(m.max()) <= 1.0:
        out = (m >= float(mask_threshold)).astype(np.uint8) * 255
        return out

    # Fallback: normalize robustly then threshold
    m_min = float(np.percentile(m, 1))
    m_max = float(np.percentile(m, 99))
    denom = (m_max - m_min) if (m_max - m_min) > 1e-6 else 1.0
    mn = np.clip((m - m_min) / denom, 0.0, 1.0)
    out = (mn >= float(mask_threshold)).astype(np.uint8) * 255
    return out


def _downscale_u8(mask_u8: np.ndarray, mask_size: int) -> np.ndarray:
    """Downscale uint8 mask so the long side == mask_size (nearest-neighbor)."""
    if mask_size <= 0:
        return mask_u8

    h, w = mask_u8.shape[:2]
    long_side = max(h, w)
    if long_side <= mask_size:
        return mask_u8

    scale = float(mask_size) / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img = Image.fromarray(mask_u8, mode="L")
    img = img.resize((new_w, new_h), resample=Image.NEAREST)
    return np.array(img, dtype=np.uint8)


def encode_mask_png_base64(mask_u8: np.ndarray) -> str:
    img = Image.fromarray(mask_u8, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============== PROMPT ENDPOINTS ==============

@app.get("/prompt")
async def get_prompt():
    return {"prompt": CURRENT_PROMPT}


@app.post("/prompt/{new_prompt:path}")
async def set_prompt(new_prompt: str):
    global CURRENT_PROMPT
    CURRENT_PROMPT = new_prompt
    print(f"✓ Prompt changed to: '{CURRENT_PROMPT}'")
    return {"prompt": CURRENT_PROMPT}


# ============== HEALTH ==============

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "prompt": CURRENT_PROMPT,
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ============== SEGMENTATION ==============

@app.post("/segment", response_model=SegmentResponse)
async def segment_image(request: SegmentRequest):
    global CURRENT_PROMPT

    prompt = request.prompt if request.prompt else CURRENT_PROMPT

    try:
        start_time = time.time()

        image = decode_image(request.image_base64)

        # Inference
        with torch.inference_mode():
            if torch.cuda.is_available():
                # FP16 autocast for speed
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    inference_state = processor.set_image(image)
                    output = processor.set_text_prompt(state=inference_state, prompt=prompt)
            else:
                inference_state = processor.set_image(image)
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        inference_time = (time.time() - start_time) * 1000.0

        if masks is None or len(masks) == 0:
            return SegmentResponse(
                success=True,
                prompt=prompt,
                num_objects=0,
                masks_base64=[],
                boxes=[],
                scores=[],
                inference_time_ms=inference_time,
            )

        masks_np = masks.detach().cpu().numpy()
        boxes_np = boxes.detach().cpu().numpy() if boxes is not None else np.zeros((len(masks_np), 4), dtype=np.float32)
        scores_np = scores.detach().cpu().numpy() if scores is not None else np.ones((len(masks_np),), dtype=np.float32)

        # Score filter
        keep = scores_np >= float(request.confidence_threshold)
        masks_np = masks_np[keep]
        boxes_np = boxes_np[keep]
        scores_np = scores_np[keep]

        if len(masks_np) == 0:
            return SegmentResponse(
                success=True,
                prompt=prompt,
                num_objects=0,
                masks_base64=[],
                boxes=[],
                scores=[],
                inference_time_ms=inference_time,
            )

        # Sort by score desc and apply top-k
        order = np.argsort(-scores_np)
        if request.max_objects > 0:
            order = order[: int(request.max_objects)]
        masks_np = masks_np[order]
        boxes_np = boxes_np[order]
        scores_np = scores_np[order]

        # Min-area filter threshold (computed from mask resolution)
        sample_u8 = _mask_to_u8(masks_np[0], request.mask_threshold)
        mh, mw = sample_u8.shape[:2]
        min_area = int(round(float(request.min_mask_area_frac) * float(mh * mw)))

        if request.mask_mode == "combined":
            combined = None
            kept_boxes = []
            kept_scores = []

            for m, b, s in zip(masks_np, boxes_np, scores_np):
                m_u8 = _mask_to_u8(m, request.mask_threshold)
                if min_area > 0 and int((m_u8 > 0).sum()) < min_area:
                    continue

                if combined is None:
                    combined = m_u8
                else:
                    combined = np.maximum(combined, m_u8)

                kept_boxes.append(b)
                kept_scores.append(s)

            if combined is None:
                return SegmentResponse(
                    success=True,
                    prompt=prompt,
                    num_objects=0,
                    masks_base64=[],
                    boxes=[],
                    scores=[],
                    inference_time_ms=inference_time,
                )

            combined = _downscale_u8(combined, int(request.mask_size))
            encoded_masks = [encode_mask_png_base64(combined)]

            return SegmentResponse(
                success=True,
                prompt=prompt,
                num_objects=len(kept_scores),
                masks_base64=encoded_masks,
                boxes=np.asarray(kept_boxes, dtype=np.float32).tolist(),
                scores=np.asarray(kept_scores, dtype=np.float32).tolist(),
                inference_time_ms=inference_time,
            )

        # instances mode (original behavior)
        encoded_masks: list[str] = []
        kept_boxes = []
        kept_scores = []

        for m, b, s in zip(masks_np, boxes_np, scores_np):
            m_u8 = _mask_to_u8(m, request.mask_threshold)
            if min_area > 0 and int((m_u8 > 0).sum()) < min_area:
                continue

            m_u8 = _downscale_u8(m_u8, int(request.mask_size))
            encoded_masks.append(encode_mask_png_base64(m_u8))
            kept_boxes.append(b)
            kept_scores.append(s)

        return SegmentResponse(
            success=True,
            prompt=prompt,
            num_objects=len(encoded_masks),
            masks_base64=encoded_masks,
            boxes=np.asarray(kept_boxes, dtype=np.float32).tolist(),
            scores=np.asarray(kept_scores, dtype=np.float32).tolist(),
            inference_time_ms=inference_time,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return SegmentResponse(
            success=False,
            prompt=prompt,
            num_objects=0,
            masks_base64=[],
            boxes=[],
            scores=[],
            inference_time_ms=0.0,
            error=str(e),
        )


if __name__ == "__main__":
    uvicorn.run("sam3_server:app", host="0.0.0.0", port=8100, reload=False, workers=1)
