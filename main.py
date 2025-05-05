from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import torch
import io
import os
import logging
import shutil
from contextlib import contextmanager
import gc
import time
from pathlib import Path

# Import your model classes
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from utils.garment_agnostic_mask_predictor import AutoMasker
from utils.densepose_predictor import DensePosePredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.getcwd(), "app.log"))
    ]
)
logger = logging.getLogger(__name__)

# Define where models will be mounted
MODEL_DIR = "/app/models"
TEMP_DIR = "/app/temp"
OUTPUT_DIR = "/app/output"

# Create necessary directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()

class ModelManager:
    def __init__(self):
        self.mask_predictor = None
        self.densepose_predictor = None
        self.vt_model = None
        self.vt_inference = None
        self.models_initialized = False
        self.initialization_error = None
        
    def initialize_models(self):
        try:
            logger.info(f"Loading models from {MODEL_DIR}")
            self.initialization_error = None
            
            # Check if model directories exist
            for directory in ["densepose", "schp", "stable-diffusion-inpainting"]:
                dir_path = os.path.join(MODEL_DIR, directory)
                if not os.path.exists(dir_path):
                    error_msg = f"Model directory {dir_path} does not exist"
                    logger.error(error_msg)
                    self.initialization_error = error_msg
                    return False
            
            # List key model files for verification
            vt_model_path = os.path.join(MODEL_DIR, "virtual_tryon.pth")
            densepose_weights = os.path.join(MODEL_DIR, "densepose", "model_final_162be9.pkl")
            densepose_config = os.path.join(MODEL_DIR, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml")
            
            if not all(os.path.exists(p) for p in [vt_model_path, densepose_weights, densepose_config]):
                error_msg = "Required model files not found in the mounted directory"
                logger.error(error_msg)
                self.initialization_error = error_msg
                return False
            
            start_time = time.time()
            
            # Initialize mask predictor
            logger.info("Initializing mask predictor...")
            self.mask_predictor = AutoMasker(
                densepose_path=os.path.join(MODEL_DIR, "densepose"),
                schp_path=os.path.join(MODEL_DIR, "schp"),
            )
            logger.info(f"Mask predictor initialized in {time.time() - start_time:.2f} seconds")

            # Initialize densepose predictor
            logger.info("Initializing densepose predictor...")
            densepose_start = time.time()
            self.densepose_predictor = DensePosePredictor(
                config_path=os.path.join(MODEL_DIR, "densepose", "densepose_rcnn_R_50_FPN_s1x.yaml"),
                weights_path=os.path.join(MODEL_DIR, "densepose", "model_final_162be9.pkl"),
            )
            logger.info(f"DensePose predictor initialized in {time.time() - densepose_start:.2f} seconds")

            # Initialize VITON model
            logger.info("Initializing VITON model...")
            viton_start = time.time()
            with torch_gc():
                model = LeffaModel(
                    pretrained_model_name_or_path=os.path.join(MODEL_DIR, "stable-diffusion-inpainting"),
                    pretrained_model=os.path.join(MODEL_DIR, "virtual_tryon.pth")
                )
                model = model.half()
                self.vt_model = model.to("cuda" if torch.cuda.is_available() else "cpu")
                self.vt_inference = LeffaInference(model=model)
            logger.info(f"VITON model initialized in {time.time() - viton_start:.2f} seconds")
            
            self.models_initialized = True
            logger.info(f"All models initialized successfully in {time.time() - start_time:.2f} seconds")
            return True
            
        except Exception as e:
            error_msg = f"Error initializing models: {str(e)}"
            logger.error(error_msg)
            self.initialization_error = error_msg
            self.models_initialized = False
            return False

    def check_model_files(self):
        """Check model files and directory structure for troubleshooting"""
        results = {}
        try:
            results["model_dir_exists"] = os.path.exists(MODEL_DIR)
            
            if not results["model_dir_exists"]:
                return results
                
            # Check directories
            directories = ["densepose", "schp", "stable-diffusion-inpainting"]
            for directory in directories:
                dir_path = os.path.join(MODEL_DIR, directory)
                results[f"{directory}_exists"] = os.path.exists(dir_path)
                
                # List files in directory
                if os.path.exists(dir_path):
                    files = os.listdir(dir_path)
                    results[f"{directory}_files"] = files[:10]  # Limit to first 10 files
                    results[f"{directory}_file_count"] = len(files)
                
            # Check main model file
            vt_model_path = os.path.join(MODEL_DIR, "virtual_tryon.pth")
            results["virtual_tryon_pth_exists"] = os.path.exists(vt_model_path)
            
            if os.path.exists(vt_model_path):
                results["virtual_tryon_pth_size"] = os.path.getsize(vt_model_path) / (1024 * 1024)  # Size in MB
                
            # Check total storage
            try:
                total, used, free = shutil.disk_usage(MODEL_DIR)
                results["disk_info"] = {
                    "total_gb": total // (2**30),
                    "used_gb": used // (2**30),
                    "free_gb": free // (2**30)
                }
            except Exception as e:
                results["disk_info_error"] = str(e)
            
        except Exception as e:
            results["error"] = str(e)
            
        return results
    
    def fallback_download_models(self):
        """Fallback function to download models if mount fails"""
        # This is a placeholder function - you would implement downloading models
        # from Azure Blob Storage if the mount fails
        logger.warning("Mount failed - downloading models is not implemented")
        return False

# Create an instance of the ModelManager
model_manager = ModelManager()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API status and model loading"""
    gpu_info = {}
    try:
        if torch.cuda.is_available():
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["cuda_version"] = torch.version.cuda
    except Exception as e:
        gpu_info["error"] = str(e)
        
    return {
        "status": "healthy",
        "models_initialized": model_manager.models_initialized,
        "initialization_error": model_manager.initialization_error,
        "model_dir_exists": os.path.exists(MODEL_DIR),
        "gpu_available": torch.cuda.is_available(),
        "gpu_info": gpu_info,
        "timestamp": time.time()
    }

# Detailed model check endpoint
@app.get("/check-models")
async def check_models():
    """Endpoint to check model files and directory structure"""
    return model_manager.check_model_files()

# Endpoint to manually initialize models
@app.post("/initialize-models")
async def initialize_models(background_tasks: BackgroundTasks):
    """Endpoint to manually initialize models"""
    # Check if models are already initialized
    if model_manager.models_initialized:
        return {"status": "success", "message": "Models already initialized"}
    
    # Start model initialization in the background
    background_tasks.add_task(model_manager.initialize_models)
    return {"status": "initializing", "message": "Model initialization started in the background"}

# Try-on endpoint
@app.post("/try-on/")
async def virtual_try_on(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    """Endpoint for virtual try-on"""
    # Check if models are initialized
    if not model_manager.models_initialized:
        init_success = model_manager.initialize_models()
        if not init_success:
            raise HTTPException(
                status_code=503, 
                detail=f"Models not initialized and initialization failed: {model_manager.initialization_error}"
            )
    
    # Generate unique filename for output
    output_filename = f"tryon_{int(time.time())}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    try:
        # Read and validate input images
        person_content = await person_image.read()
        garment_content = await garment_image.read()
        
        if len(person_content) == 0 or len(garment_content) == 0:
            raise HTTPException(status_code=400, detail="Empty image file uploaded")
        
        # Convert uploaded files to PIL Images
        try:
            person_img = Image.open(io.BytesIO(person_content))
            garment_img = Image.open(io.BytesIO(garment_content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # Convert to RGB if needed
        if person_img.mode != 'RGB':
            person_img = person_img.convert('RGB')
        if garment_img.mode != 'RGB':
            garment_img = garment_img.convert('RGB')

        # Log image sizes
        logger.info(f"Processing: person image {person_img.size}, garment image {garment_img.size}")

        # Process images for VITON
        start_time = time.time()
        try:
            with torch_gc():
                # Get mask and parse
                logger.info("Getting mask and parse...")
                mask_start = time.time()
                mask_output = model_manager.mask_predictor(person_img)
                agnostic = mask_output["agnostic"]
                parse = mask_output["parse"]
                logger.info(f"Mask generation completed in {time.time() - mask_start:.2f} seconds")
                
                # Get dense pose
                logger.info("Getting dense pose...")
                dense_start = time.time()
                dense_output = model_manager.densepose_predictor(person_img)
                dense = dense_output["dense"]
                logger.info(f"DensePose generation completed in {time.time() - dense_start:.2f} seconds")

                # Perform virtual try-on
                logger.info("Performing virtual try-on...")
                tryon_start = time.time()
                output = model_manager.vt_inference.infer(
                    source_image=person_img,
                    target_image=garment_img,
                    agnostic=agnostic,
                    parse=parse,
                    dense=dense
                )
                logger.info(f"Try-on completed in {time.time() - tryon_start:.2f} seconds")

                # Save the result
                output.save(output_path)
                logger.info(f"Total processing time: {time.time() - start_time:.2f} seconds")

                # Return the result
                return FileResponse(output_path, media_type="image/png", filename=output_filename)
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        # Clean up
        clear_memory()

# Application startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models on application startup"""
    logger.info("Application starting up...")
    try:
        logger.info("Checking model files...")
        model_files_check = model_manager.check_model_files()
        logger.info(f"Model files check: {model_files_check}")
        
        logger.info("Initializing models on startup...")
        init_success = model_manager.initialize_models()
        
        if init_success:
            logger.info("Models initialized successfully on startup")
        else:
            logger.error(f"Failed to initialize models on startup: {model_manager.initialization_error}")
            # We don't raise the exception here to allow the API to start
            # Models can be loaded later via the /initialize-models endpoint
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown"""
    logger.info("Application shutting down...")
    clear_memory()
