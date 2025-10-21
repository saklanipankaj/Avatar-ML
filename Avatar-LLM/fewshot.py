"""fewshot.py

Utilities to select few-shot examples and run classification via the Bedrock-backed
DefectDetectionModel.predict_fewshot method.
"""
import os
import random
from helper import get_directory_path, get_image_files
from model import DefectDetectionModel
from config import DEFECT_PROMPTS


def select_few_shot_examples(defect_name, k=3, include_labels=True):
    """Select k example images for defect_name. Returns list of (label, path) or paths.

    include_labels: if True, the returned list items will be (label, path), where
    label is 'defect' or 'non-defect' inferred from filename if possible.
    """
    directory = get_directory_path(defect_name)
    if not directory or not os.path.exists(directory):
        return []

    files = get_image_files(directory)
    if not files:
        return []

    # Random sample k examples
    k = min(k, len(files))
    sampled = random.sample(files, k)

    results = []
    for f in sampled:
        path = os.path.join(directory, f)
        label = None
        if include_labels:
            # Heuristic: filenames containing 'non' are non-defect
            if 'non' in f.lower():
                label = 'non-defect'
            else:
                label = 'defect'

        results.append((label, path) if include_labels else path)

    return results


def classify_with_few_shot(defect_name, query_image_path, k=3, include_labels=True):
    """Run few-shot classification for a single query image.

    - Select k examples from the defect_name directory
    - Append the query image (without label)
    - Build the prompt from DEFECT_PROMPTS[defect_name]
    - Call model.predict_fewshot and return the raw model text
    """
    prompt = DEFECT_PROMPTS.get(defect_name)
    if not prompt:
        raise ValueError(f"No prompt for defect type: {defect_name}")

    examples = select_few_shot_examples(defect_name, k=k, include_labels=include_labels)

    # Append the query image as unlabeled last item
    items = examples + [query_image_path]

    model = DefectDetectionModel()
    result = model.predict_fewshot(items, prompt)
    return result
