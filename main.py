from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import numpy as np
from typing import Optional
from pydantic import BaseModel
from fastapi.responses import Response
import torch
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc
from leffa_utils.densepose_predictor import DensePosePredictor


class MaskingService:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
            device=device
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )
        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx"
        )
        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth"
        )

    def resize_and_center(self, image: Image.Image, height: int, width: int) -> Image.Image:
        """Resize and center the image while maintaining aspect ratio"""
        aspect_ratio = image.size[0] / image.size[1]
        if aspect_ratio > width / height:
            new_width = width
            new_height = int(width / aspect_ratio)
        else:
            new_height = height
            new_width = int(height * aspect_ratio)
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create new image with padding
        new_image = Image.new('RGB', (width, height), (255, 255, 255))
        paste_x = (width - new_width) // 2
        paste_y = (height - new_height) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image

    def get_mask(self, image: Image.Image, garment_type: str = "upper_body") -> Image.Image:
        """Generate mask for the given image and garment type"""
        # Resize image for processing
        image = self.resize_and_center(image, 768, 1024)
        image = image.convert("RGB")
        
        # Get parsing and keypoints
        model_parse, _ = self.parsing(image.resize((384, 512)))
        keypoints = self.openpose(image.resize((384, 512)))
        
        # Generate mask based on garment type
        mask = get_agnostic_mask_hd(model_parse, keypoints, garment_type)
        mask = mask.resize((768, 1024))
        
        return mask

app = FastAPI(title="Virtual Try-On Masking API")

@app.post("/generate_mask")
async def generate_mask(
    file: UploadFile = File(...),
    garment_type: str = "upper_body"
):
    """
    Generate a mask for virtual try-on
    
    Args:
        file: Image file to process
        garment_type: Type of garment ("upper_body", "lower_body", or "dresses")
    
    Returns:
        PNG image of the generated mask
    """
    if garment_type not in ["upper_body", "lower_body", "dresses"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid garment type. Must be 'upper_body', 'lower_body', or 'dresses'"
        )
    
    try:
        # Read and validate image
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # Initialize service if not already initialized
        if not hasattr(app, "masking_service"):
            app.masking_service = MaskingService()
        
        # Generate mask
        mask = app.masking_service.get_mask(image, garment_type)
        
        # Convert mask to bytes
        img_byte_arr = io.BytesIO()
        mask.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Startup event to initialize the masking service
@app.on_event("startup")
async def startup_event():
    app.masking_service = MaskingService()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)