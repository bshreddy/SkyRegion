from flask import Flask, request, send_file, jsonify
import io

import model

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome. This is a Flask AI Inference Server"

@app.route('/predict', methods=['POST'])
def predict():
    img_bytes = request.files["img"].read()
    mask = model.get_mask_image(img_bytes)
    return serve_pil_image(mask)

@app.route('/maoe', methods=['POST'])
def getMAOEs():
    img_bytes = request.files["img"].read()
    maoe = model.get_maoe(img_bytes)
    return jsonify(maoe)

def serve_pil_image(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png', attachment_filename="mask.png")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000)