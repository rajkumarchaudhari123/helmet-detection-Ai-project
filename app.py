from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import cv2
import numpy as np
import base64
import uuid
from datetime import datetime
import logging
import glob
import torch
import traceback

app = Flask(__name__)

# ================= CONFIG =================
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= MODEL LOAD =================
try:
    model_path = r"C:\runs\detect\train\weights\best.pt"  # r prefix = raw string

    print("Checking model path:", model_path)
    print("Exists:", os.path.exists(model_path))

    if os.path.exists(model_path):
        model = YOLO(model_path)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        CLASS_NAMES = {
            0: "✅ Helmet",
            1: "❌ No Helmet"
        }

        logger.info(f"🔥 Model loaded on {device}")

    else:
        model = None
        CLASS_NAMES = {}
        logger.error("❌ Model file not found!")

except Exception as e:
    logger.error(f"❌ Model load error: {str(e)}")
    model = None
    CLASS_NAMES = {}

# ================= COLORS =================
COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255)
}

# ================= CLEANUP =================
def cleanup_old_files():
    files = sorted(glob.glob('static/results/*.jpg'), key=os.path.getmtime)
    if len(files) > 50:
        for f in files[:-50]:
            os.remove(f)

# ================= IMAGE PROCESS =================
def process_image(img):
    try:
        print("Processing image...")

        img = cv2.resize(img, (640, 640))
        results = model(img, conf=0.4)

        detections = []
        helmet_count = 0
        no_helmet_count = 0

        for r in results:
            boxes = r.boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if cls == 0:
                        helmet_count += 1
                    elif cls == 1:
                        no_helmet_count += 1

                    color = COLORS.get(cls, (255, 255, 255))
                    label = f"{CLASS_NAMES.get(cls)} ({conf:.2f})"

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)

                    detections.append({
                        "class": CLASS_NAMES.get(cls),
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2]
                    })

        if no_helmet_count > 0:
            message = f"❌ {no_helmet_count} No Helmet Detected"
        elif helmet_count > 0:
            message = f"✅ {helmet_count} Helmet Detected"
        else:
            message = "🤔 No Detection"

        filename = f"result_{uuid.uuid4()}.jpg"
        path = os.path.join(app.config['RESULT_FOLDER'], filename)
        cv2.imwrite(path, img)

        cleanup_old_files()

        return {
            "detections": detections,
            "helmet": helmet_count,
            "no_helmet": no_helmet_count,
            "message": message,
            "image": f"/static/results/{filename}",
            "time": datetime.now().strftime("%H:%M:%S")
        }

    except Exception as e:
        print("PROCESS ERROR:", str(e))
        traceback.print_exc()
        return None

# ================= ROUTES =================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-camera', methods=['POST'])
def predict_camera():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image received'}), 400

        if ',' in image_data:
            image_data = image_data.split(',')[1]

        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Image decode failed'}), 400

        result = process_image(img)

        if result is None:
            return jsonify({'error': 'Processing failed'}), 500

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        print("API ERROR:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

# ================= RUN =================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
