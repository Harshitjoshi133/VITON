from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
from fastapi.responses import Response
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.utils import get_agnostic_mask_hd
from leffa_utils.densepose_predictor import DensePosePredictor
import nest_asyncio
from pyngrok import ngrok
import uvicorn

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

    def get_mask(self, image: Image.Image, garment_type: str = "upper_body") -> Image.Image:
        image = image.convert("RGB")
        model_parse, _ = self.parsing(image.resize((384, 512)))
        keypoints = self.openpose(image.resize((384, 512)))
        mask = get_agnostic_mask_hd(model_parse, keypoints, garment_type)
        return mask.resize((768, 1024))

app = FastAPI(title="Virtual Try-On Masking API")

@app.post("/generate_mask")
async def generate_mask(file: UploadFile = File(...), garment_type: str = "upper_body"):
    if garment_type not in ["upper_body", "lower_body", "dresses"]:
        raise HTTPException(status_code=400, detail="Invalid garment type")
    
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        if not hasattr(app, "masking_service"):
            app.masking_service = MaskingService()
        mask = app.masking_service.get_mask(image, garment_type)
        img_byte_arr = io.BytesIO()
        mask.save(img_byte_arr, format='PNG')
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    app.masking_service = MaskingService()

def start_ngrok():
    port = 8000
    public_url = ngrok.connect(port).public_url
    print(f"Public URL: {public_url}")

if __name__ == "__main__":
    nest_asyncio.apply()
    start_ngrok()
    uvicorn.run(app, host="0.0.0.0", port=8000)