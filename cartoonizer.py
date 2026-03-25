"""
cartoonizer.py
--------------
A simple image cartoonizer I built for my Computer Vision project.

The idea is pretty straightforward — take a real photo and make it
look like a hand-drawn cartoon. I'm doing this in 4 stages:

  1. Smooth out the image without killing the edges (bilateral filter)
  2. Pull out the strong edges and turn them into bold outlines
  3. Flatten the colors down to a small palette (like a comic book)
  4. Slap the outlines on top of the flat colors

It's not magic — just a few classic CV tricks chained together nicely.

Author : vidhi udasi 23bai10202
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
    I'm using K-Means clustering here to reduce the image down to
    a fixed number of colors. Think of it like posterization —
    instead of millions of shades, we pick N representative colors
    and repaint every pixel with whichever one is closest.

    This gives that flat, illustrated look you see in cartoons.

    img        : the image we want to simplify (BGR numpy array)
    num_colors : how many colors to keep (the K in K-Means)

    returns    : a new image with flattened/simplified colors
    """

    h, w = img.shape[:2]

    # K-Means needs a flat list of pixels in float32 format
    pixel_list = img.reshape((-1, 3)).astype(np.float32)

    # We tell it to stop after 20 rounds or when improvement is tiny
    stop_condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)

    _, cluster_ids, palette = cv2.kmeans(
        pixel_list,
        num_colors,
        None,
        stop_condition,
        attempts=8,
        flags=cv2.KMEANS_RANDOM_CENTERS
    )

    # Replace each pixel with the color of its cluster center
    palette = np.uint8(palette)
    flat_pixels = palette[cluster_ids.flatten()]

    return flat_pixels.reshape((h, w, 3))




def smooth_image(img, strength):
    """
    Bilateral filtering is my go-to here because unlike Gaussian blur,
    it doesn't bleed across hard edges. It looks at both how far apart
    pixels are AND how different their colors are before deciding
    whether to blend them.

    I apply it a few times in a loop to get a more exaggerated,
    painterly smoothness — one pass isn't usually enough.

    img      : original image
    strength : controls how aggressively colors get blended together

    returns  : smoothed image
    """

    result = img.copy()
    for _ in range(4):
        result = cv2.bilateralFilter(result, d=9, sigmaColor=strength, sigmaSpace=strength)

    return result



def get_edges(img):
    """
    To get those bold dark outlines, I convert to grayscale first,
    then use adaptive thresholding. Unlike regular thresholding,
    adaptive mode calculates the cutoff separately for small regions
    of the image — so it handles photos with uneven lighting much better.

    The result is a black-and-white mask where white = edge, black = not.
    I then invert it so edges become dark lines (more cartoon-like).

    img     : smoothed image to extract edges from

    returns : binary edge mask (3-channel, so it blends with color image)
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold picks a local cutoff per region
    edge_mask = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=9,   # size of local neighborhood
        C=2            # small constant subtracted from the local mean
    )

    # Convert to 3 channels so we can AND it with the color image
    return cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)




def run_cartoonizer(input_path, output_path, num_colors=8, strength=300):
    """
    Ties everything together. The steps go like this:

        original → smooth → flat colors
                          → edges
                          → combine (AND)  → cartoon

    input_path  : path to the photo you want to cartoonize
    output_path : where to save the result
    num_colors  : number of flat colors in the final cartoon
    strength    : how aggressively the bilateral filter smooths things

    returns     : output_path on success
    raises      : FileNotFoundError or ValueError if something's wrong
    """

    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Couldn't find the image at: '{input_path}'")

    original = cv2.imread(input_path)
    if original is None:
        raise ValueError(
            f"OpenCV couldn't open '{input_path}'. "
            "Check that it's a valid image file (JPG, PNG, BMP, WEBP)."
        )

    h, w = original.shape[:2]
    print(f"  Loaded image  →  {w} x {h} pixels")

    
    print("  Step 1  →  Smoothing with bilateral filter ...")
    smoothed = smooth_image(original, strength)

    print("  Step 2  →  Extracting cartoon edges ...")
    edges = get_edges(smoothed)


    print(f"  Step 3  →  Flattening colors down to {num_colors} shades ...")
    flat = flatten_colors(smoothed, num_colors)

    print("  Step 4  →  Merging edges onto flat colors ...")
    cartoon = cv2.bitwise_and(flat, edges)

    
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(output_path, cartoon)
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cartoonizer",
        description="Turn any photo into a cartoon using OpenCV.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cartoonizer.py selfie.jpg out.jpg\n"
            "  python cartoonizer.py selfie.jpg out.jpg --colors 6\n"
            "  python cartoonizer.py selfie.jpg out.jpg --colors 10 --strength 150\n"
        )
    )

    parser.add_argument("input",  help="Input image path (JPG / PNG / BMP / WEBP)")
    parser.add_argument("output", help="Where to save the cartoon output")

    parser.add_argument(
        "--colors",
        type=int,
        default=8,
        metavar="N",
        help="How many flat colors to use (default: 8)\n"
             "Try 4-6 for abstract, 10-12 for more detail"
    )
    parser.add_argument(
        "--strength",
        type=int,
        default=300,
        metavar="N",
        help="Smoothing strength for bilateral filter (default: 300)\n"
             "Higher = softer/smoother look"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print()
    print("=" * 46)
    print("     CartoonLens  —  Image Cartoonizer")
    print("=" * 46)
    print(f"  Input    :  {args.input}")
    print(f"  Output   :  {args.output}")
    print(f"  Colors   :  {args.colors}")
    print(f"  Strength :  {args.strength}")
    print("-" * 46)

    try:
        result = run_cartoonizer(
            input_path=args.input,
            output_path=args.output,
            num_colors=args.colors,
            strength=args.strength
        )
        print("-" * 46)
        print(f"  Done!  Saved cartoon →  {result}")
        print("=" * 46)
        print()

    except FileNotFoundError as e:
        print(f"\n  [Error]  {e}\n")
        sys.exit(1)

    except ValueError as e:
        print(f"\n  [Error]  {e}\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n  [Unexpected Error]  {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
