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

# Get the absolute path of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= MODEL LOAD =================
try:
    # Try multiple possible locations for the model
    possible_paths = [
        # Your original path
        r"C:\runs\detect\train\weights\best.pt",
        # Project models folder
        os.path.join(BASE_DIR, "models", "best.pt"),
        # Current directory
        os.path.join(BASE_DIR, "best.pt"),
    ]
    
    model_path = None
    for path in possible_paths:
        print(f"Checking: {path} - Exists: {os.path.exists(path)}")
        if os.path.exists(path):
            model_path = path
            break
    
    print("=" * 50)
    print("Current working directory:", os.getcwd())
    print("Base directory:", BASE_DIR)
    print("Selected model path:", model_path)
    print("=" * 50)

    if model_path:
        model = YOLO(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        CLASS_NAMES = {
            0: "✅ Helmet",
            1: "❌ No Helmet"
        }
        
        logger.info(f"🔥 Model loaded successfully on {device}")
        print(f"✅ Model loaded from: {model_path}")
        
    else:
        model = None
        CLASS_NAMES = {}
        logger.error("❌ Model file not found!")
        print("❌ ERROR: Could not find best.pt in any of these locations:")
        for i, path in enumerate(possible_paths, 1):
            print(f"   {i}. {path}")

except Exception as e:
    logger.error(f"❌ Model load error: {str(e)}")
    print(f"❌ Exception during model load: {str(e)}")
    traceback.print_exc()
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
            try:
                os.remove(f)
            except:
                pass

# ================= IMAGE PROCESS =================
def process_image(img):
    try:
        if model is None:
            print("❌ Model not loaded, cannot process image")
            return None
            
        print("Processing image...")
        
        # Resize image
        img = cv2.resize(img, (640, 640))
        
        # Run inference
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
                    label = f"{CLASS_NAMES.get(cls, 'Unknown')} ({conf:.2f})"

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label background
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(img, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(img, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)

                    detections.append({
                        "class": CLASS_NAMES.get(cls, "Unknown"),
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2]
                    })

        # Create message
        if no_helmet_count > 0:
            message = f"❌ {no_helmet_count} No Helmet Detected"
        elif helmet_count > 0:
            message = f"✅ {helmet_count} Helmet Detected"
        else:
            message = "🤔 No Detection"

        # Save result image
        filename = f"result_{uuid.uuid4()}.jpg"
        path = os.path.join(app.config['RESULT_FOLDER'], filename)
        cv2.imwrite(path, img)

        # Cleanup old files
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
            error_msg = "Model not loaded. Please check if model file exists."
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 500

        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image received'}), 400

        # Remove base64 header if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode image
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Image decode failed'}), 400

        # Process image
        result = process_image(img)

        if result is None:
            return jsonify({'error': 'Processing failed'}), 500

        return jsonify({
            "success": True,
            "data": result,
            "results": {
                "helmet_detected": result["helmet"] > 0,
                "no_helmet_detected": result["no_helmet"] > 0,
                "result_message": result["message"],
                "result_path": result["image"],
                "total": len(result["detections"]),
                "timestamp": result["time"],
                "detections": result["detections"]
            }
        })

    except Exception as e:
        print("API ERROR:", str(e))
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Enhanced health check endpoint"""
    # Check all possible paths
    possible_paths = [
        r"C:\runs\detect\train\weights\best.pt",
        os.path.join(BASE_DIR, "models", "best.pt"),
        os.path.join(BASE_DIR, "best.pt"),
    ]
    
    path_status = {}
    for path in possible_paths:
        path_status[str(path)] = os.path.exists(path)
    
    model_status = {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "model_path": model_path if 'model_path' in locals() else None,
        "paths_checked": path_status,
        "cuda_available": torch.cuda.is_available(),
        "current_directory": os.getcwd(),
        "files_in_models": os.listdir('models') if os.path.exists('models') else [],
        "static_folders": {
            "uploads_exists": os.path.exists(app.config['UPLOAD_FOLDER']),
            "results_exists": os.path.exists(app.config['RESULT_FOLDER'])
        }
    }
    
    if model is not None:
        model_status["device"] = str(model.device)
    
    return jsonify(model_status)

# ================= RUN =================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Starting Helmet Detection App")
    print("="*50)
    print(f"📍 Base directory: {BASE_DIR}")
    print("📍 Checking for model in:")
    for path in possible_paths:
        print(f"   - {path}: {'✅ Found' if os.path.exists(path) else '❌ Not found'}")
    print(f"📍 Model loaded: {model is not None}")
    print(f"📍 CUDA available: {torch.cuda.is_available()}")
    print("="*50 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)