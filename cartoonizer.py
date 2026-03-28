"""
cartoonizer.py
--------------

This script converts a normal image into a cartoon-style version using OpenCV.

The approach is simple:
- Smooth the image but keep edges visible
- Detect edges to create outlines
- Reduce colors to a limited palette
- Combine everything into a cartoon effect

Author : Vidhi Udasi (23BAI10202)
Course : Computer Vision — VIT Bhopal University
Date   : March 2026

Usage:
    python cartoonizer.py input.jpg output.jpg
    python cartoonizer.py input.jpg output.jpg --colors 6
    python cartoonizer.py input.jpg output.jpg --colors 10 --strength 150
"""

import cv2
import numpy as np
import argparse
import os
import sys


def flatten_colors(img, num_colors):
    """
    Reduces the number of colors using K-Means clustering.

    The image is grouped into a fixed number of color clusters.
    Each pixel is replaced with its cluster center color.

    img        : input image
    num_colors : number of colors to keep

    returns    : simplified image
    """

    h, w = img.shape[:2]

    pixels = img.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        0.5
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        num_colors,
        None,
        criteria,
        8,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    result = centers[labels.flatten()]

    return result.reshape((h, w, 3))


def smooth_image(img, strength):
    """
    Applies bilateral filtering to smooth the image.

    This keeps edges intact while reducing noise.
    Running it multiple times increases the effect.

    img      : input image
    strength : smoothing intensity

    returns  : smoothed image
    """

    result = img.copy()

    for _ in range(4):
        result = cv2.bilateralFilter(
            result,
            d=9,
            sigmaColor=strength,
            sigmaSpace=strength
        )

    return result


def get_edges(img):
    """
    Extracts edges using adaptive thresholding.

    The image is converted to grayscale first.
    Adaptive threshold helps handle uneven lighting.

    img     : input image

    returns : edge mask (3-channel)
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        2
    )

    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def run_cartoonizer(input_path, output_path, num_colors=8, strength=300):
    """
    Runs the full pipeline to generate the cartoon image.

    input_path  : input image path
    output_path : output image path
    num_colors  : number of colors
    strength    : smoothing level

    returns     : output path
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Couldn't find the image: {input_path}")

    original = cv2.imread(input_path)

    if original is None:
        raise ValueError("Invalid image file")

    h, w = original.shape[:2]
    print(f"  Loaded image → {w} x {h}")

    print("  Step 1 → Smoothing...")
    smoothed = smooth_image(original, strength)

    print("  Step 2 → Detecting edges...")
    edges = get_edges(smoothed)

    print(f"  Step 3 → Reducing colors to {num_colors}...")
    flat = flatten_colors(smoothed, num_colors)

    print("  Step 4 → Combining...")
    cartoon = cv2.bitwise_and(flat, edges)

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(output_path, cartoon)

    return output_path


def build_parser():
    """
    Creates command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="cartoonizer",
        description="Convert an image into cartoon style"
    )

    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")

    parser.add_argument(
        "--colors",
        type=int,
        default=8,
        help="Number of colors (default: 8)"
    )

    parser.add_argument(
        "--strength",
        type=int,
        default=300,
        help="Smoothing strength (default: 300)"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("\n" + "=" * 40)
    print("   CartoonLens — Image Cartoonizer")
    print("=" * 40)
    print(f"Input    : {args.input}")
    print(f"Output   : {args.output}")
    print(f"Colors   : {args.colors}")
    print(f"Strength : {args.strength}")
    print("-" * 40)

    try:
        result = run_cartoonizer(
            args.input,
            args.output,
            args.colors,
            args.strength
        )

        print("-" * 40)
        print(f"Saved → {result}")
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"\nError: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
