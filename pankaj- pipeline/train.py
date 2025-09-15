import argparse
import os
import logging
import sys
import subprocess # Needed if building detectron2 from source

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances # Or Cityscapes specific registration
from detectron2.evaluation import COCOEvaluator # Or CityscapesEvaluator

# Add other necessary Detectron2 imports
print(f"Detectron2 version: {detectron2.__version__}")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_OUTPUT_DATA_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
SM_CHANNEL_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation') # If using validation channel
SM_NUM_GPUS = int(os.environ.get('SM_NUM_GPUS', 10)) # Get number of GPUs from SageMaker

def register_cityscapes_datasets(dataset_dir, train_name):
    """
    Registers the Cityscapes dataset in Detectron2 format.
    Assumes data is structured correctly in dataset_dir.
    Modify this based on your actual data format (e.g., COCO format).
    """
    logger.info(f"Registering dataset '{train_name}' from directory: {dataset_dir}")
    # Example using register_coco_instances if data is in COCO format
    # Adjust json_file and image_root paths accordingly
    train_json_file = os.path.join(dataset_dir, "annotations", "instancesonly_filtered_gtFine_train.json") # Example path
    train_image_root = os.path.join(dataset_dir, "leftImg8bit", "train") # Example path

    if not os.path.exists(train_json_file) or not os.path.exists(train_image_root):
         logger.warning(f"Training data/annotations not found at expected paths:")
         logger.warning(f" JSON: {train_json_file}")
         logger.warning(f" Images: {train_image_root}")
         logger.warning("Skipping dataset registration. Ensure data is correctly placed in the input channel.")
         # Depending on requirements, you might want to raise an error here
         # raise FileNotFoundError("Cityscapes training data not found.")
         return False # Indicate registration failed

    try:
        register_coco_instances(train_name, {}, train_json_file, train_image_root)
        logger.info(f"Successfully registered dataset: {train_name}")
        # Optionally register validation set if needed for evaluation during training
        # val_json_file = ...
        # val_image_root = ...
        # register_coco_instances("cityscapes_fine_instance_seg_val", {}, val_json_file, val_image_root)
        return True # Indicate registration succeeded
    except Exception as e:
        logger.error(f"Failed to register dataset '{train_name}': {e}")
        return False


def setup_cfg(args):
    """
    Creates and configures Detectron2's CfgNode based on arguments.
    """
    cfg = get_cfg()

    # Load Base Model Config
    try:
        # Append CfgNode attributes for custom arguments used in training logic
        cfg.merge_from_file(model_zoo.get_config_file(args.model_config_name))
        logger.info(f"Loaded base config from: {args.model_config_name}")
    except FileNotFoundError:
        logger.error(f"Cannot find base model config: {args.model_config_name}")
        raise
    except KeyError:
         logger.error(f"Model config '{args.model_config_name}' not found in model zoo.")
         raise

    # Load Pre-trained Weights
    # Load COCO pre-trained weights, fine-tuning on Cityscapes
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.model_config_name)
    logger.info(f"Loading pre-trained weights from: {cfg.MODEL.WEIGHTS}")

    # Dataset Configuration
    cfg.DATASETS.TRAIN = (args.dataset_train_name,)
    cfg.DATASETS.TEST = () # No test set during training, use separate eval step

    # Data Loader
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    # Solver Configuration (Hyperparameters)
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch * SM_NUM_GPU # Scale batch size by number of GPUs
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []  # Do not decay learning rate (as per user's config example)
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period # How often to save checkpoints

    # Model Configuration
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.roi_batch_size_per_image
    cfg.MODEL.BACKBONE.FREEZE_AT = args.freeze_at

    # Output Directory
    # Checkpoints and logs saved within the job's container
    cfg.OUTPUT_DIR = "/opt/ml/output/checkpoints"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger.info(f"Detectron2 output directory set to: {cfg.OUTPUT_DIR}")

    # Finalize Config
    cfg.freeze()
    return cfg

# Main Training Function
def main(args):
    """
    Main training loop.
    """
    # Register Datasets
    # Assumes training data is in the 'training' channel directory
    dataset_registered = register_cityscapes_datasets(SM_CHANNEL_TRAINING, args.dataset_train_name)
    if not dataset_registered:
        logger.error("Dataset registration failed. Exiting.")
        sys.exit(1)

    # Setup Configuration
    try:
        cfg = setup_cfg(args)
    except Exception as e:
         logger.error(f"Failed to setup Detectron2 config: {e}")
         sys.exit(1)

    # Setup Trainer
    # Use default setup for logging, seeding, etc. based on cfg
    default_setup(cfg, args) # args here are parsed command-line args, not the hyperparameter dict

    # Check if resuming from a checkpoint within the job's output dir
    # SageMaker handles job restarts, but Detectron2 needs the resume flag
    resume_training = args.resume # Use the hyperparameter value

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume_training)
    logger.info(f"Trainer initialized. Resuming training: {resume_training}")

    # Add Hooks (Optional)
    # Example: Add evaluation hook if validation set is registered and needed during training
    # if cfg.DATASETS.TEST: # If a test/validation set is configured
    #     evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=os.path.join(cfg.OUTPUT_DIR, "eval"))
    #     trainer.register_hooks([hooks.EvalHook(cfg.TEST.EVAL_PERIOD, lambda: evaluator)])

    # Start Training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training finished successfully.")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Optionally save state or logs before exiting
        raise # Re-raise exception to mark SageMaker job as failed

    # Save Final Model
    # The DefaultTrainer saves checkpoints periodically to cfg.OUTPUT_DIR.
    # We need to copy the final desired model artifact to SM_MODEL_DIR.
    # Typically, this is the last checkpoint saved.
    final_model_path_d2 = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # Default name by Detectron2

    if os.path.exists(final_model_path_d2):
        # SageMaker expects the model artifact in SM_MODEL_DIR
        # Copy the final PyTorch model state_dict
        final_model_dest = os.path.join(SM_MODEL_DIR, "model.pth") # Standard name for SageMaker PyTorch
        logger.info(f"Copying final model from {final_model_path_d2} to {final_model_dest}")
        # Using shutil for potentially large files
        import shutil
        shutil.copy2(final_model_path_d2, final_model_dest)

        # Optionally, save the config file used for this training run alongside the model
        config_save_path = os.path.join(SM_MODEL_DIR, "config.yaml")
        logger.info(f"Saving training config to {config_save_path}")
        with open(config_save_path, "w") as f:
            f.write(cfg.dump()) # Save the configuration used

        logger.info("Model and config saved to SM_MODEL_DIR.")
    else:
        logger.error(f"Final model 'model_final.pth' not found in {cfg.OUTPUT_DIR}. Cannot save to SM_MODEL_DIR.")
        # Decide if this is a critical error
        sys.exit("Failed to find final model artifact.")


# Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments expected from SageMaker estimator hyperparameters
    parser.add_argument("--model-config-name", type=str, required=True, help="Detectron2 base model config file name")
    parser.add_argument("--dataset-train-name", type=str, required=True, help="Name for registering the training dataset")
    parser.add_argument("--learning-rate", type=float, default=0.00025, help="Base learning rate")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum training iterations")
    parser.add_argument("--ims-per-batch", type=int, default=2, help="Images per batch per GPU")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--checkpoint-period", type=int, default=500, help="Save checkpoint every N iterations")
    parser.add_argument("--freeze-at", type=int, default=2, help="Freeze backbone layers up to this stage")
    parser.add_argument("--roi-batch-size-per-image", type=int, default=256, help="RoI Heads batch size per image")
    parser.add_argument("--resume", type=lambda x: (str(x).lower() == 'true'), default=False, help="Resume training from last checkpoint")

    # Add standard Detectron2 arguments
    # These might overlap with hyperparameters but are standard for launch utility
    parser.add_argument("--num-gpus", type=int, default=SM_NUM_GPUS, help="Number of GPUs to use (obtained from SM environment)")
    parser.add_argument("--num-machines", type=int, default=1, help="Number of machines (nodes)")
    parser.add_argument("--machine-rank", type=int, default=0, help="Rank of the current machine")
    parser.add_argument("--dist-url", type=str, default="auto", help="URL for distributed training setup")
    # Add config-file and opts for compatibility if needed, though we set cfg manually
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # Parse Arguments
    # Detectron2's launch utility might parse args differently,
    # but we parse them here to configure cfg based on hyperparameters.
    args = parser.parse_args()

    logger.info("Starting training script with arguments:")
    logger.info(args)
    logger.info(f"Running with {SM_NUM_GPUS} GPUs.")

    # Launch Training
    launch(
        main, # The main function to execute
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,), # Pass parsed args tuple to main function
    )
