#!/usr/bin/env python3
"""
Monocular Depth Estimation Server

Provides depth estimation from single RGB images using state-of-the-art models.
Supports multiple backends:
  - Depth Anything V2 (default, best quality)
  - MiDaS (fallback)
  - Simple heuristic (no ML)

Usage:
    # With Depth Anything V2 (recommended)
    conda activate depth  # or your ML environment
    python mono_depth_server.py --model depth_anything_v2
    
    # With MiDaS
    python mono_depth_server.py --model midas
    
    # Simple mode (no ML, for testing)
    python mono_depth_server.py --model simple

API Endpoints:
    GET  /health     - Server status
    POST /estimate   - Estimate depth from image
    GET  /info       - Model information
"""

import argparse
import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# ============== Model backends (loaded conditionally) ==============
model = None
model_type = None
transform = None
device = None


class DepthRequest(BaseModel):
    image_base64: str
    # Output options
    output_size: Optional[int] = 0       # 0 = same as input, else resize
    normalize: bool = True               # Normalize depth to 0-1 range
    metric_scale: float = 10.0           # Scale factor for metric depth (meters)


class DepthResponse(BaseModel):
    success: bool
    depth_base64: Optional[str] = None   # Base64 encoded float32 numpy array
    width: int = 0
    height: int = 0
    min_depth: float = 0.0
    max_depth: float = 0.0
    inference_time_ms: float = 0.0
    model: str = ""
    error: Optional[str] = None


def decode_image(base64_str: str) -> Image.Image:
    """Decode base64 image to PIL Image"""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")


def load_depth_anything_v2():
    """Load Depth Anything V2 model"""
    global model, transform, device
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Try to load Depth Anything V2
    try:
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        
        model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        print(f"Loading {model_name}...")
        
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
        transform = AutoImageProcessor.from_pretrained(model_name)
        
        model = model.to(device).eval()
        
        if device.type == "cuda":
            model = model.half()  # FP16 for speed
        
        print(f"✓ Depth Anything V2 loaded on {device}")
        return True
        
    except ImportError:
        print("transformers not installed, trying alternative...")
        return False
    except Exception as e:
        print(f"Failed to load Depth Anything V2: {e}")
        return False


def load_midas():
    """Load MiDaS model"""
    global model, transform, device
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load MiDaS from torch hub
        print("Loading MiDaS DPT-Hybrid...")
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform
        
        model = model.to(device).eval()
        
        print(f"✓ MiDaS loaded on {device}")
        return True
        
    except Exception as e:
        print(f"Failed to load MiDaS: {e}")
        return False


def load_simple():
    """Load simple heuristic (no ML model)"""
    global model, transform
    model = "simple"
    transform = None
    print("✓ Simple depth heuristic loaded (no ML)")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    global model_type
    
    print("=" * 60)
    print("Mono Depth Server Starting...")
    print("=" * 60)
    
    args = getattr(app, 'args', None) or argparse.Namespace(model='auto')
    
    success = False
    
    if args.model == 'auto':
        # Try models in order of preference
        if load_depth_anything_v2():
            model_type = "depth_anything_v2"
            success = True
        elif load_midas():
            model_type = "midas"
            success = True
        else:
            load_simple()
            model_type = "simple"
            success = True
            
    elif args.model == 'depth_anything_v2':
        success = load_depth_anything_v2()
        model_type = "depth_anything_v2" if success else None
        
    elif args.model == 'midas':
        success = load_midas()
        model_type = "midas" if success else None
        
    elif args.model == 'simple':
        success = load_simple()
        model_type = "simple"
    
    if not success:
        print("WARNING: No depth model loaded!")
        model_type = None
    
    print("=" * 60)
    print(f"✓ Server ready at http://0.0.0.0:8101")
    print(f"✓ Model: {model_type}")
    print("=" * 60)
    
    yield
    
    # Cleanup
    global model, transform
    model = None
    transform = None
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Mono Depth Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "no_model",
        "model": model_type,
        "cuda_available": False,  # Will be updated below
    }


@app.get("/info")
async def info():
    """Model information"""
    import torch
    return {
        "model": model_type,
        "device": str(device) if device else "cpu",
        "cuda_available": torch.cuda.is_available() if 'torch' in dir() else False,
        "supported_models": ["depth_anything_v2", "midas", "simple"],
    }


def estimate_depth_anything_v2(image: Image.Image) -> np.ndarray:
    """Run Depth Anything V2 inference"""
    import torch
    
    # Preprocess
    inputs = transform(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    if device.type == "cuda":
        inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth
    
    # Post-process
    depth = depth.squeeze().cpu().numpy()
    
    # Resize to original size
    depth_pil = Image.fromarray(depth)
    depth_pil = depth_pil.resize(image.size, Image.BILINEAR)
    
    return np.array(depth_pil)


def estimate_midas(image: Image.Image) -> np.ndarray:
    """Run MiDaS inference"""
    import torch
    import cv2
    
    # Convert to numpy
    img_np = np.array(image)
    
    # Preprocess
    input_batch = transform(img_np).to(device)
    
    # Inference
    with torch.no_grad():
        depth = model(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth = depth.cpu().numpy()
    
    # MiDaS outputs inverse depth, so we invert
    depth = 1.0 / (depth + 1e-6)
    
    return depth


def estimate_simple(image: Image.Image) -> np.ndarray:
    """
    Simple depth heuristic based on vertical position.
    Objects lower in the image are assumed to be closer.
    This is a rough approximation and should only be used for testing.
    """
    w, h = image.size
    
    # Create gradient depth map (bottom = close, top = far)
    y_coords = np.linspace(0, 1, h).reshape(h, 1)
    depth = np.tile(y_coords, (1, w))
    
    # Invert so bottom is closer (smaller depth)
    depth = 1.0 - depth
    
    # Scale to reasonable depth range (0.5 to 10 meters)
    depth = depth * 9.5 + 0.5
    
    return depth.astype(np.float32)


@app.post("/estimate", response_model=DepthResponse)
async def estimate_depth(request: DepthRequest):
    """Estimate depth from RGB image"""
    global model, model_type
    
    if model is None:
        return DepthResponse(
            success=False,
            error="No depth model loaded"
        )
    
    try:
        start_time = time.time()
        
        # Decode image
        image = decode_image(request.image_base64)
        orig_size = image.size
        
        # Run inference based on model type
        if model_type == "depth_anything_v2":
            depth = estimate_depth_anything_v2(image)
        elif model_type == "midas":
            depth = estimate_midas(image)
        else:  # simple
            depth = estimate_simple(image)
        
        # Convert relative depth to metric (approximate)
        if request.normalize:
            # Normalize to 0-1 range
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth = (depth - d_min) / (d_max - d_min)
            
            # Scale to metric depth
            depth = depth * request.metric_scale
        
        # Resize output if requested
        if request.output_size > 0:
            depth_pil = Image.fromarray(depth)
            
            # Calculate new size maintaining aspect ratio
            w, h = orig_size
            if w > h:
                new_w = request.output_size
                new_h = int(h * request.output_size / w)
            else:
                new_h = request.output_size
                new_w = int(w * request.output_size / h)
            
            depth_pil = depth_pil.resize((new_w, new_h), Image.BILINEAR)
            depth = np.array(depth_pil)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Encode depth as base64
        depth_bytes = depth.astype(np.float32).tobytes()
        depth_b64 = base64.b64encode(depth_bytes).decode('utf-8')
        
        return DepthResponse(
            success=True,
            depth_base64=depth_b64,
            width=depth.shape[1],
            height=depth.shape[0],
            min_depth=float(depth.min()),
            max_depth=float(depth.max()),
            inference_time_ms=inference_time,
            model=model_type,
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return DepthResponse(
            success=False,
            error=str(e),
            model=model_type or "none"
        )


# ============== Visualization endpoint (optional) ==============

@app.post("/visualize")
async def visualize_depth(request: DepthRequest):
    """Get depth as colorized visualization image"""
    import cv2
    
    # First get depth estimation
    result = await estimate_depth(request)
    
    if not result.success:
        return result
    
    # Decode depth
    depth_bytes = base64.b64decode(result.depth_base64)
    depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(result.height, result.width)
    
    # Normalize for visualization
    d_min, d_max = depth.min(), depth.max()
    depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', depth_colored, [cv2.IMWRITE_JPEG_QUALITY, 90])
    vis_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "success": True,
        "visualization_base64": vis_b64,
        "width": result.width,
        "height": result.height,
        "min_depth": result.min_depth,
        "max_depth": result.max_depth,
    }


def main():
    parser = argparse.ArgumentParser(description="Monocular Depth Estimation Server")
    parser.add_argument(
        "--model", 
        type=str, 
        default="auto",
        choices=["auto", "depth_anything_v2", "midas", "simple"],
        help="Depth estimation model to use"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8101)
    
    args = parser.parse_args()
    
    # Store args for lifespan to access
    app.args = args
    
    uvicorn.run(
        "mono_depth_server:app",
        host=args.host,
        port=args.port,
        reload=False,
        workers=1
    )


if __name__ == "__main__":
    main()
