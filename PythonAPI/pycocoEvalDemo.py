#!/usr/bin/env python3
"""
COCO Evaluation Demo

This script demonstrates how to use the COCO API for evaluating object detection,
segmentation, and keypoint detection results.
"""

# %%
# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import pylab

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import _mask as maskUtils

# Configure matplotlib
pylab.rcParams['figure.figsize'] = (10.0, 8.0)




# %%
def configure_evaluation_parameters():
    """Configure evaluation parameters for the demo."""
    # Available annotation types
    annotation_types = ['segm', 'bbox', 'keypoints']    
    # Select annotation type (0: segm, 1: bbox, 2: keypoints)
    selected_type = annotation_types[1]  # Using bbox for this demo
    # Determine prefix based on annotation type
    prefix = 'person_keypoints' if selected_type == 'keypoints' else 'instances'
    
    print(f'Running demo for *{selected_type}* results.')
    
    return selected_type, prefix


# %%
def initialize_coco_apis(ann_file, results_file):
    """Initialize COCO ground truth and detection APIs."""
    print(f"Loading ground truth annotations from: {ann_file}")
    coco_gt = COCO(ann_file)
    
    print(f"Loading detection results from: {results_file}")
    coco_dt = coco_gt.loadRes(results_file)
    
    return coco_gt, coco_dt


# %%
def prepare_evaluation_data(coco_gt, num_images=100):
    """Prepare image IDs for evaluation."""
    # Get all image IDs and select a subset
    img_ids = sorted(coco_gt.getImgIds())
    img_ids = img_ids[:num_images]
    
    # Select a random image for potential visualization
    random_img_id = img_ids[np.random.randint(len(img_ids))]
    
    print(f"Using {len(img_ids)} images for evaluation")
    print(f"Random image ID selected: {random_img_id}")
    
    return img_ids, random_img_id


# %%
def run_coco_evaluation(coco_gt, coco_dt, ann_type, img_ids):
    """Run COCO evaluation and display results."""
    print("Starting COCO evaluation...")
    
    # Initialize evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
    coco_eval.params.imgIds = img_ids
    
    # Run evaluation pipeline
    print("Evaluating...")
    coco_eval.evaluate()
    
    print("Accumulating...")
    coco_eval.accumulate()
    
    print("Summarizing results...")
    coco_eval.summarize()
    
    return coco_eval


# %%
def main():
    """Main function to run the COCO evaluation demo."""
    # Verify environment setup
    
    # Configure parameters
    annotation_types = ['segm', 'bbox', 'keypoints']    
    ann_type = annotation_types[1]  # Using bbox for this demo
    prefix = 'person_keypoints' if ann_type == 'keypoints' else 'instances'
    
    # Set up file paths
    data_dir = '../'
    data_type = 'val2014'
    ann_file = 'data/instances_val2014.json'
    results_file = f'{data_dir}/results/{prefix}_{data_type}_fake{ann_type}100_results.json'
    
    # Initialize COCO APIs
    coco_gt, coco_dt = initialize_coco_apis(ann_file, results_file)
    
    # Prepare evaluation data
    img_ids, random_img_id = prepare_evaluation_data(coco_gt)
    
    # Run evaluation
    coco_eval = run_coco_evaluation(coco_gt, coco_dt, ann_type, img_ids)
    
    print("Evaluation completed successfully!")
    return coco_eval


# %%
# Run the demo
if __name__ == "__main__" or "get_ipython" in globals():
    coco_eval_result = main()


