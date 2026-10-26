# AI Project 4 — Image or Text Recognition (Basic)

A small Flask web application inspired by the supplied Artificial Intelligence Project 4 slides. It offers two paths:

1. **OCR (Optical Character Recognition):** preprocesses an uploaded image with OpenCV and extracts text with Tesseract.
2. **Object detection:** detects common objects using the Caffe MobileNet-SSD model and draws bounding boxes with confidence scores.

## Requirements
- Python 3.10 or newer
- Tesseract OCR engine for the OCR option
- MobileNet-SSD model files for the object-detection option

## Run on Windows
Open Command Prompt in this folder:

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Install Tesseract for OCR
Install the Tesseract OCR engine for Windows (for example, from the UB Mannheim Tesseract distribution). If it is not on PATH, set its executable path before starting Flask:

```bat
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

### Enable object detection (optional)
Create the `models` folder if needed and place these files inside it:
- `MobileNetSSD_deploy.prototxt`
- `MobileNetSSD_deploy.caffemodel`

The commonly used original MobileNet-SSD Caffe files are available from the `chuanqi305/MobileNet-SSD` GitHub repository: `https://github.com/chuanqi305/MobileNet-SSD`. Download the prototxt and caffemodel from that repository and save them with the exact filenames above. Model files are not included in this ZIP.

### Start the app

```bat
python app.py
```

Open `http://127.0.0.1:5000` in your browser. Use **OCR** if you want text extracted from a document/photo, or **Object Detection** to locate supported objects such as people, cars, bicycles, buses, dogs, and bottles.

## Notes
- Maximum upload size is 12 MB. Supported image types: PNG, JPG/JPEG, BMP, WEBP.
- Object detection uses a pretrained model; it does not train a new model.
- OCR and object detection are local after dependencies/models are installed. Google Fonts in the UI may require internet access; the application itself runs on localhost.
- This is an educational starter project. Results depend on image quality and model limitations.
