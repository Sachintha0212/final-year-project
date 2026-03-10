from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  
UPLOAD_FOLDER = "uploads"  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  

@app.route("/health", methods=["GET"]) 
def health():
    return jsonify({"status": "running"})  


@app.route("/predict", methods=["POST"])  
def predict():
   
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Use form-data key: image"}), 400

    img = request.files["image"]
    save_path = os.path.join(UPLOAD_FOLDER, filename) 
    img.save(save_path)

    # TODO: Replace this with your ML model prediction
    # result = your_model_predicts(save_paths)
    result = {
        "class": "sample_prediction",  
        "confidence": 0.88,  
        "file_saved": filename  
    }
    return jsonify(result)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 