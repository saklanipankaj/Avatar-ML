"""run_fewshot.py

Simple runner to perform few-shot image classification.

Usage: edit variables below or import and call from other scripts.
"""
import os
import argparse
from fewshot import classify_with_few_shot


def main():
    parser = argparse.ArgumentParser(description="Few-shot image classification runner")
    parser.add_argument("defect", help="Defect type (e.g. frosted_window)")
    parser.add_argument("image", help="Path to query image")
    parser.add_argument("-k", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--out", help="Output file to save model response (optional)")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Query image not found: {args.image}")
        return

    print(f"Running few-shot classification for '{args.defect}' on {args.image} with k={args.k}")
    resp = classify_with_few_shot(args.defect, args.image, k=args.k)

    print("\nModel Response:\n")
    print(resp)

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(resp)
        print(f"Saved response to {args.out}")


if __name__ == '__main__':
    main()
