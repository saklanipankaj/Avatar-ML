import argparse
import os
import json
import logging
import sys
import torch # PyTorch needed to load the model

# Detectron2 Imports
import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances # Or Cityscapes specific
from detectron2.evaluation import COCOEvaluator, inference_on_dataset # Or CityscapesEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Setup Logging
setup_logger()
logger = logging.getLogger("detectron2") # Use Detectron2's logger

# Constants
# Paths within the processing job container
MODEL_DIR = "/opt/ml/processing/model"
INPUT_DATA_DIR = "/opt/ml/processing/input_data"
# EVAL_DATA_DIR = "/opt/ml/processing/eval_data" # If using separate eval data input
OUTPUT_DIR = "/opt/ml/processing/evaluation"


def register_eval_dataset(dataset_dir, dataset_name):
    """Registers the dataset needed for evaluation."""
    logger.info(f"Registering evaluation dataset '{dataset_name}' from directory: {dataset_dir}")
    # Adapt this based on your evaluation dataset format (e.g., COCO, Cityscapes)
    # Example assumes COCO format, using the 'val' split conventionally
    json_file = os.path.join(dataset_dir, "annotations", "instancesonly_filtered_gtFine_val.json") # Example path
    image_root = os.path.join(dataset_dir, "leftImg8bit", "val") # Example path

    if not os.path.exists(json_file) or not os.path.exists(image_root):
         logger.warning(f"Evaluation data/annotations not found at expected paths:")
         logger.warning(f" JSON: {json_file}")
         logger.warning(f" Images: {image_root}")
         logger.warning("Skipping dataset registration for evaluation.")
         return False

    try:
        register_coco_instances(dataset_name, {}, json_file, image_root)
        logger.info(f"Successfully registered dataset: {dataset_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to register dataset '{dataset_name}': {e}")
        return False

def load_model_and_config(model_dir, args):
    """Loads the trained model and its configuration."""
    config_path = os.path.join(model_dir, "config.yaml")
    model_path = os.path.join(model_dir, "model.pth")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        logger.error(f"Model ('model.pth') or config ('config.yaml') not found in {model_dir}")
        raise FileNotFoundError("Model or config file missing.")

    logger.info(f"Loading config from: {config_path}")
    cfg = get_cfg()
    # Load config saved during training
    cfg.merge_from_file(config_path)

    # Crucial adjustments for inference/evaluation
    # Load the specific weights saved by the training job
    cfg.MODEL.WEIGHTS = model_path
    # Set score threshold for detections (adjust as needed)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    # Ensure dataset name for testing is set if needed by evaluator
    # cfg.DATASETS.TEST = (args.dataset_eval_name,) # Set if evaluator uses it

    cfg.freeze()
    logger.info("Configuration loaded and adjusted for evaluation.")

    # Build Model
    model = build_model(cfg)
    logger.info("Model built.")

    # Load Weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    logger.info(f"Loaded model weights from {cfg.MODEL.WEIGHTS}")
    model.eval() # Set model to evaluation mode

    return model, cfg


# Main Evaluation Function
def main(args):
    logger.info("Starting evaluation script.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Register evaluation dataset
    # Use the same name as specified in job arguments, load from input data dir
    eval_dataset_name = args.dataset_eval_name # e.g., "cityscapes_fine_instance_seg_val"
    dataset_registered = register_eval_dataset(INPUT_DATA_DIR, eval_dataset_name)
    if not dataset_registered:
         logger.error("Evaluation dataset registration failed. Cannot perform evaluation.")
         # Save an empty/error report
         report = {"evaluation_status": "failed", "error": "Dataset registration failed"}
         report_path = os.path.join(OUTPUT_DIR, "evaluation.json")
         with open(report_path, "w") as f:
             json.dump(report, f)
         sys.exit(1)


    # Load Model and Config
    try:
        model, cfg = load_model_and_config(MODEL_DIR, args)
    except FileNotFoundError:
        logger.error("Failed to load model or config. Exiting.")
        report = {"evaluation_status": "failed", "error": "Model/config loading failed"}
        report_path = os.path.join(OUTPUT_DIR, "evaluation.json")
        with open(report_path, "w") as f:
            json.dump(report, f)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during model loading: {e}")
        report = {"evaluation_status": "failed", "error": f"Model loading error: {e}"}
        report_path = os.path.join(OUTPUT_DIR, "evaluation.json")
        with open(report_path, "w") as f:
            json.dump(report, f)
        sys.exit(1)


    # Setup Evaluator
    # Choose the evaluator based on the dataset (COCO, Cityscapes, etc.)
    # Ensure the dataset name matches the one registered
    evaluator_type = "coco" # Default to COCO, change if using Cityscapes specific evaluator
    if "cityscapes" in eval_dataset_name.lower():
        # from detectron2.evaluation import CityscapesInstanceEvaluator # If available and needed
        # evaluator = CityscapesInstanceEvaluator(eval_dataset_name)
        # evaluator_type = "cityscapes"
        # Using COCO evaluator for Cityscapes in COCO format is common
        evaluator = COCOEvaluator(eval_dataset_name, output_dir=OUTPUT_DIR)
        evaluator_type = "coco"
        logger.info(f"Using COCOEvaluator for dataset: {eval_dataset_name}")
    else:
        evaluator = COCOEvaluator(eval_dataset_name, output_dir=OUTPUT_DIR)
        logger.info(f"Using COCOEvaluator for dataset: {eval_dataset_name}")

    # Run Inference and Evaluation
    logger.info(f"Running inference on dataset: {eval_dataset_name}")
    try:
        # Get the data loader for the registered evaluation dataset
        data_loader = detectron2.data.build_detection_test_loader(cfg, eval_dataset_name)
        results = inference_on_dataset(model, data_loader, evaluator)
        logger.info("Inference and evaluation completed.")
        logger.info(f"Evaluation results: {results}")

        # Prepare Evaluation Report for SageMaker
        # Extract key metrics (e.g., mAP) from the results dictionary
        # The structure of 'results' depends on the evaluator used (COCOEvaluator typically returns dict with 'segm' key)
        report_metrics = {}
        if evaluator_type == "coco" and "segm" in results:
            # Extract standard COCO mAP metrics
            for k, v in results["segm"].items():
                # Ensure metric names are simple strings for JSON
                metric_name = f"segmentation_{k.replace('/', '_')}" # Replace '/' if present
                report_metrics[metric_name] = v
        else:
             # Handle other evaluator types or if results structure is different
             report_metrics = {"custom_results": results} # Save raw results if structure unknown

        # Add overall status
        report = {"evaluation_status": "success", "metrics": report_metrics}

    except Exception as e:
        logger.error(f"Error during evaluation inference: {e}")
        report = {"evaluation_status": "failed", "error": f"Inference/Evaluation error: {e}"}


    # Save Evaluation Report
    # Save the report as evaluation.json in the output directory
    report_path = os.path.join(OUTPUT_DIR, "evaluation.json")
    logger.info(f"Saving evaluation report to: {report_path}")
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info("Evaluation report saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save evaluation report: {e}")
        # Still try to exit cleanly, but log the error
        pass

# Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add arguments if needed by the evaluation script (e.g., specific thresholds)
    parser.add_argument("--model-config-name", type=str, required=True, help="Detectron2 base model config name (used for context)")
    parser.add_argument("--dataset-eval-name", type=str, default="cityscapes_fine_instance_seg_val", help="Name the evaluation dataset is registered under")
    # Add other arguments as needed

    args = parser.parse_args()
    main(args)