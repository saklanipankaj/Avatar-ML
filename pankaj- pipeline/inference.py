import os
import json
import logging
import sys
import io # For handling image bytes

import torch
import numpy as np
from PIL import Image

# Detectron2 Imports
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
print(f"Detectron2 version: {detectron2.__version__}")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# SageMaker Inference Toolkit Functions
def model_fn(model_dir):
    """
    Loads the model and configuration.
    Called once when the SageMaker endpoint container starts.
    """
    logger.info(f"Loading model from {model_dir}...")
    config_path = os.path.join(model_dir, "config.yaml")
    model_path = os.path.join(model_dir, "model.pth")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        logger.error(f"Model ('model.pth') or config ('config.yaml') not found in {model_dir}")
        raise FileNotFoundError("Model or config file missing in model directory.")

    # Load Config
    cfg = get_cfg()
    cfg.merge_from_file(config_path)

    # Adjustments for Inference
    cfg.MODEL.WEIGHTS = model_path # Point to the loaded model file
    # Set device based on availability (GPU preferred)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {cfg.MODEL.DEVICE}")
    # Set thresholds (can be overridden in predict_fn if needed)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # Example threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5 # Example NMS threshold

    cfg.freeze()
    logger.info("Configuration loaded and adjusted for inference.")

    # Create Predictor
    try:
        predictor = DefaultPredictor(cfg)
        logger.info("Detectron2 DefaultPredictor created successfully.")
    except Exception as e:
        logger.error(f"Failed to create DefaultPredictor: {e}")
        raise
		
    metadata = None
    if cfg.DATASETS.TRAIN:
         try:
            dataset_name = cfg.DATASETS.TRAIN[0] # Get dataset name used for training
            metadata = MetadataCatalog.get(dataset_name)
            logger.info(f"Loaded metadata for dataset: {dataset_name}")
         except KeyError:
             logger.warning(f"Metadata for dataset '{cfg.DATASETS.TRAIN[0]}' not found in MetadataCatalog. Visualization might lack class names.")
         except Exception as e:
             logger.warning(f"Error getting metadata: {e}")

    model_data = {'predictor': predictor, 'cfg': cfg, 'metadata': metadata}
    return model_data


def input_fn(request_body, request_content_type):
    """
    Deserializes the input request body.
    Handles different content types (e.g., image/jpeg, application/x-npy).
    """
    logger.info(f"Received request with Content-Type: {request_content_type}")
    if request_content_type == 'image/jpeg' or request_content_type == 'image/png':
        try:
            img_bytes = io.BytesIO(request_body)
            image = Image.open(img_bytes).convert('RGB') # Ensure RGB format
            # Convert PIL Image to NumPy array (BGR format expected by Detectron2)
            # OpenCV reads as BGR, PIL reads as RGB. Convert RGB -> BGR.
            np_image = np.array(image)[:, :, ::-1]
            logger.info(f"Image loaded and converted to NumPy array (shape: {np_image.shape})")
            return np_image
        except Exception as e:
            logger.error(f"Failed to process image input: {e}")
            raise ValueError(f"Could not decode image input: {e}")
    elif request_content_type == 'application/x-npy':
        try:
            np_image = np.load(io.BytesIO(request_body))
            logger.info(f"NumPy array loaded (shape: {np_image.shape})")
            # Assume input numpy array is already in BGR format if needed
            return np_image
        except Exception as e:
            logger.error(f"Failed to process NumPy input: {e}")
            raise ValueError(f"Could not load NumPy array: {e}")
    else:
        raise ValueError(f"Unsupported Content-Type: {request_content_type}")


def predict_fn(input_data, model_data):
    """
    Performs inference using the loaded model and preprocessed input.
    """
    logger.info("Performing prediction...")
    predictor = model_data['predictor']
    cfg = model_data['cfg']
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # Run Prediction
    try:
        with torch.no_grad():
            outputs = predictor(input_data) # input_data is the NumPy array (BGR)
        logger.info("Prediction successful.")
        # 'outputs' contains predictions in Detectron2 format (instances, etc.)
        # logger.debug(f"Raw prediction output keys: {outputs.keys()}")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

    # Visualize Output (Combine mask and original image)
    logger.info("Visualizing prediction output...")
    try:
        # input_data is BGR, Visualizer expects BGR by default
        v = Visualizer(input_data[:, :, ::-1], metadata=metadata, scale=1.2)
        # Draw predictions on the image
        if "instances" in outputs:
            out_vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            # Get the visualized image as a NumPy array (RGB format)
            visualized_image_rgb = out_vis.get_image()
            logger.info("Visualization complete.")
            return visualized_image_rgb # Return the visualized image (NumPy array, RGB)
        else:
            logger.warning("No 'instances' found in predictor output. Returning original image.")
            # Return original image (converted back to RGB) if no instances found
            return input_data[:, :, ::-1] # Convert BGR back to RGB

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        # Return the original image (RGB) as fallback if visualization fails
        return input_data[:, :, ::-1] # Convert BGR back to RGB


def output_fn(prediction_output, response_content_type):
    """
    Serializes the prediction output.
    Handles the requested response content type (e.g., image/jpeg).
    """
    logger.info(f"Serializing prediction output for Content-Type: {response_content_type}")
    if response_content_type == 'image/jpeg' or response_content_type == 'image/png':
        try:
            # prediction_output is the visualized NumPy array (RGB)
            image = Image.fromarray(prediction_output.astype(np.uint8))
            img_byte_arr = io.BytesIO()
            format = 'JPEG' if response_content_type == 'image/jpeg' else 'PNG'
            image.save(img_byte_arr, format=format)
            img_byte_arr = img_byte_arr.getvalue()
            logger.info(f"Prediction serialized successfully as {format}.")
            return img_byte_arr
        except Exception as e:
            logger.error(f"Failed to serialize image output: {e}")
            raise ValueError(f"Could not serialize image output: {e}")
    else:
        raise ValueError(f"Unsupported Accept type: {response_content_type}")

