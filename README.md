# CartoonLens   Image Cartoonizer

A Computer Vision command-line tool that converts any real photograph into classic cartoon-style art using OpenCV.  
Built as part of the **Bring Your Own Project (BYOP)** component for the flipped course at VIT Bhopal University.

---

## What This Project Does

Most photos have millions of colors and soft gradients. Cartoons don't — they have bold outlines and flat blocks of color.  
This tool bridges that gap by running a photo through 4 computer vision steps:

1. **Bilateral Filtering** — Smooths the image repeatedly without blurring the edges (unlike Gaussian blur)
2. **Adaptive Thresholding** — Detects strong edges locally and turns them into bold dark outlines
3. **K-Means Color Quantization** — Clusters all pixel colors into a small fixed palette (like a comic book)
4. **Bitwise AND** — Overlays the edge mask on the flat-color image to produce the final cartoon

The result looks hand-drawn, with clean outlines and simplified flat colors.

---

## Computer Vision Concepts Covered

| Concept | Where Used |
|---|---|
| Bilateral Filtering | Smoothing step — preserves edges while reducing noise |
| Grayscale Conversion | Pre-processing before edge detection |
| Adaptive Thresholding | Edge extraction with local brightness handling |
| K-Means Clustering | Color quantization / palette reduction |
| Bitwise Operations | Combining edge mask with color image |
| NumPy Array Manipulation | Pixel reshaping and cluster mapping |

---

## Requirements

- Python **3.8 or higher**
- pip (comes with Python)
- A terminal / command prompt

No GUI or display setup needed — everything runs from the command line.

---

## Environment Setup

### Step 1 — Make sure Python is installed

```bash
python --version
```

You should see something like `Python 3.10.x`. If not, download it from https://www.python.org/downloads/

---

### Step 2 — Clone the repository

```bash
git clone //https://github.com/vidhi-sys/Cartoon-Image-Creation-By-OPENCV.git
cd image-cartoonizer
```

---

### Step 3 — Create a virtual environment (recommended)

This keeps dependencies isolated from your system Python.

**On Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You'll know it's active when your terminal prompt shows `(venv)` at the start.

---

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` — core image processing library
- `numpy` — array operations used internally by OpenCV

**If you're on Google Colab**, use this instead:
```bash
pip install opencv-python-headless numpy
```

---

## Running the Project

### Basic usage

```bash
python cartoonizer.py <input_image> <output_image>
```

**Example:**
```bash
python cartoonizer.py photo.jpg cartoon.jpg
```

---

### Advanced usage — tuning the output

```bash
python cartoonizer.py <input_image> <output_image> --colors N --strength N
```

| Flag | Default | What it does |
|---|---|---|
| `--colors N` | `8` | Number of flat colors in the cartoon. Lower = more abstract, Higher = more detailed |
| `--strength N` | `300` | Smoothing intensity of the bilateral filter. Higher = softer, painterly look |

**Examples:**

```bash
# More abstract / comic-book look
python cartoonizer.py photo.jpg cartoon.jpg --colors 5

# More detailed / natural cartoon
python cartoonizer.py photo.jpg cartoon.jpg --colors 12 --strength 150

# Strong smoothing with few colors
python cartoonizer.py photo.jpg cartoon.jpg --colors 6 --strength 400
```

---

### Supported input formats

JPG, JPEG, PNG, BMP, WEBP

---

## Expected Terminal Output

When the script runs successfully, you will see:

```
==============================================
     CartoonLens  —  Image Cartoonizer
==============================================
  Input    :  photo.jpg
  Output   :  cartoon.jpg
  Colors   :  8
  Strength :  300
----------------------------------------------
  Loaded image  →  1024 x 768 pixels
  Step 1  →  Smoothing with bilateral filter ...
  Step 2  →  Extracting cartoon edges ...
  Step 3  →  Flattening colors down to 8 shades ...
  Step 4  →  Merging edges onto flat colors ...
----------------------------------------------
  Done!  Saved cartoon →  cartoon.jpg
==============================================
```

The cartoonized image will be saved at the path you specified.

---

## Project Structure

```
image-cartoonizer/
├── cartoonizer.py       # Main CLI script — all logic lives here
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

### What each file does

**`cartoonizer.py`**  
Contains 4 functions:
- `smooth_image()` — applies bilateral filter multiple times
- `get_edges()` — grayscale conversion + adaptive thresholding
- `flatten_colors()` — K-Means clustering for color reduction
- `run_cartoonizer()` — main pipeline that calls all the above
- `main()` — CLI entry point with argument parsing

**`requirements.txt`**  
Lists the two external libraries needed: `opencv-python` and `numpy`.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`**  
→ Run `pip install opencv-python` (or `opencv-python-headless` on Colab)

**`FileNotFoundError: Couldn't find the image`**  
→ Double-check the input path and filename. Make sure the image is in the same folder or provide the full path.

**Output image looks too dark or mostly black**  
→ Try increasing `--colors` to 10 or 12, and lowering `--strength` to 150

**Output image looks too blurry**  
→ Lower `--strength` to 100–150 for sharper edges

**`ValueError: OpenCV couldn't open the file`**  
→ The file might be corrupted or in an unsupported format. Try converting it to JPG first.

---

## How to Verify It Works (Google Colab)

```python
# Step 1 — Install
!pip install opencv-python-headless numpy

# Step 2 — Upload cartoonizer.py and a test image
from google.colab import files
uploaded = files.upload()

# Step 3 — Run
!python cartoonizer.py photo.jpg cartoon.jpg

# Step 4 — View result
from IPython.display import display
from PIL import Image
display(Image.open("cartoon.jpg"))

# Step 5 — Download
files.download("cartoon.jpg")
```

---

## Author

**Name:** vidhi udasi  
**Registration No:** 23BAI10202 
**Course:** Computer Vision  
**Institution:** VIT Bhopal University, Madhya Pradesh — 466114  
**Submission:** BYOP — VITyarthi Portal  
**Date:** March 2026
