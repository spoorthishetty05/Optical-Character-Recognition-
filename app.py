
"""Project 4: Image or Text Recognition (Basic) - Flask web application."""

from pathlib import Path
import os
import uuid

import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)
from werkzeug.utils import secure_filename

# EasyOCR setup
try:
    import easyocr
    reader = None
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    reader = None
    EASYOCR_AVAILABLE = False

# Project directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
RESULT_DIR = BASE_DIR / "static" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_CONTENT_LENGTH = 12 * 1024 * 1024

app = Flask(__name__)
app.secret_key = os.environ.get(
    "SECRET_KEY", "change-this-development-key"
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)


# MobileNet-SSD object detection configuration
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

PROTOTXT = BASE_DIR / "models" / "MobileNetSSD_deploy.prototxt"
MODEL = BASE_DIR / "models" / "MobileNetSSD_deploy.caffemodel"


def allowed_file(filename):
    """Check whether the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_ocr_reader():
    global reader

    if not EASYOCR_AVAILABLE:
        raise RuntimeError(
            "EasyOCR is not installed. Run: "
            "python -m pip install easyocr"
        )

    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False)

    return reader


def run_ocr(image_path):
    """Recognize printed text in an uploaded image using EasyOCR."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            "Could not read this image. Please try another image."
        )

    ocr_reader = get_ocr_reader()
    results = ocr_reader.readtext(str(image_path))

    recognized_lines = []

    for result in results:
        if len(result) >= 3:
            detected_text = result[1].strip()
            confidence = float(result[2])

            if detected_text:
                recognized_lines.append(
                    f"{detected_text} ({confidence * 100:.1f}%)"
                )

    if recognized_lines:
        text = "\n".join(recognized_lines)
    else:
        text = "No readable text was detected in this image."

    return text, image


def run_object_detection(
    image_path, output_path, confidence_threshold=0.50
):
    """Detect supported objects using the MobileNet-SSD model."""
    if not PROTOTXT.exists() or not MODEL.exists():
        raise RuntimeError(
            "MobileNet-SSD model files are missing. Place "
            "MobileNetSSD_deploy.prototxt and "
            "MobileNetSSD_deploy.caffemodel in the models folder."
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            "Could not read this image. Please try another image."
        )

    height, width = image.shape[:2]

    net = cv2.dnn.readNetFromCaffe(
        str(PROTOTXT), str(MODEL)
    )

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    found = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence < confidence_threshold:
            continue

        class_id = int(detections[0, 0, i, 1])

        if class_id <= 0 or class_id >= len(CLASSES):
            continue

        box = (
            detections[0, 0, i, 3:7]
            * np.array([width, height, width, height])
        )

        x1, y1, x2, y2 = box.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        label = CLASSES[class_id]
        percentage = round(confidence * 100, 1)

        found.append({
            "label": label,
            "confidence": percentage
        })

        caption = f"{label}: {percentage:.1f}%"

        cv2.rectangle(
            image, (x1, y1), (x2, y2), (35, 130, 230), 2
        )

        cv2.putText(
            image,
            caption,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (35, 130, 230),
            2
        )

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(
            "Could not save the annotated result image."
        )

    return found


@app.route("/", methods=["GET"])
def index():
    """Display the upload page."""
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():
    """Handle uploaded images and run the selected recognition mode."""
    file = request.files.get("image")
    mode = request.form.get("mode", "ocr")

    if not file or not file.filename:
        flash("Please choose an image first.", "error")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash(
            "Unsupported file type. Upload PNG, JPG, JPEG, BMP, or WEBP.",
            "error"
        )
        return redirect(url_for("index"))

    if mode not in {"ocr", "detect"}:
        flash("Please select a valid recognition mode.", "error")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    stem, ext = os.path.splitext(filename)
    token = uuid.uuid4().hex[:10]

    saved_name = f"{stem[:60]}_{token}{ext.lower()}"
    source_path = UPLOAD_DIR / saved_name

    file.save(source_path)

    try:
        if mode == "detect":
            result_name = f"result_{Path(saved_name).stem}.jpg"
            result_path = RESULT_DIR / result_name

            detections = run_object_detection(
                source_path, result_path
            )

            return render_template(
                "result.html",
                mode="detect",
                original=saved_name,
                result=result_name,
                detections=detections,
                text=""
            )

        text, _ = run_ocr(source_path)

        return render_template(
            "result.html",
            mode="ocr",
            original=saved_name,
            result=None,
            detections=[],
            text=text
        )

    except Exception as exc:
        app.logger.exception("Recognition failed")
        flash(str(exc), "error")
        return redirect(url_for("index"))


@app.errorhandler(413)
def too_large(_error):
    """Handle files larger than the upload limit."""
    flash(
        "Image is too large. Maximum upload size is 12 MB.",
        "error"
    )
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
