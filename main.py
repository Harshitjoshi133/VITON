from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import torch
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from utils.garment_agnostic_mask_predictor import AutoMasker
from utils.densepose_predictor import DensePosePredictor
import io
import os
from contextlib import contextmanager
import gc
import nest_asyncio
from pyngrok import ngrok
import uvicorn

app = FastAPI(title="VITON API", description="Virtual Try-On API using Leffa")

# Memory management context manager
@contextmanager
def torch_gc():
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
            with torch.cuda.device('cuda'):
                torch.cuda.empty_cache()

def clear_memory():
    gc.collect()

class ModelManager:
    def __init__(self):
        self.mask_predictor = None
        self.densepose_predictor = None
        self.vt_model = None
        self.vt_inference = None
        self.initialize_models()

    def initialize_models(self):
        # Initialize mask predictor
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        # Initialize densepose predictor
        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        # Initialize VITON model
        with torch_gc():
            model = LeffaModel(
                pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
                pretrained_model="./ckpts/virtual_tryon.pth"
            )
            model = model.half()
            self.vt_model = model.to("cuda")
            self.vt_inference = LeffaInference(model=model)

model_manager = ModelManager()

@app.post("/try-on/")
async def virtual_try_on(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    try:
        # Convert uploaded files to PIL Images
        person_img = Image.open(io.BytesIO(await person_image.read()))
        garment_img = Image.open(io.BytesIO(await garment_image.read()))

        # Convert to RGB if needed
        if person_img.mode != 'RGB':
            person_img = person_img.convert('RGB')
        if garment_img.mode != 'RGB':
            garment_img = garment_img.convert('RGB')

        # Process images for VITON
        with torch_gc():
            # Get mask and parse
            mask_output = model_manager.mask_predictor(person_img)
            agnostic = mask_output["agnostic"]
            parse = mask_output["parse"]
            
            # Get dense pose
            dense_output = model_manager.densepose_predictor(person_img)
            dense = dense_output["dense"]

            # Perform virtual try-on
            output = model_manager.vt_inference.infer(
                source_image=person_img,
                target_image=garment_img,
                agnostic=agnostic,
                parse=parse,
                dense=dense
            )

            # Save the result temporarily
            output_path = "temp_output.png"
            output.save(output_path)

            # Return the result
            return FileResponse(output_path, media_type="image/png", filename="try_on_result.png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up
        clear_memory()
        if os.path.exists("temp_output.png"):
            os.remove("temp_output.png")

def start_ngrok():
    port = 8000
    public_url = ngrok.connect(port).public_url
    print(f"Public URL: {public_url}")

if __name__ == "__main__":
    nest_asyncio.apply()
    start_ngrok()
    uvicorn.run(app, host="0.0.0.0", port=8000)
