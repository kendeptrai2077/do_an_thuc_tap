from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import json
import os

app = Flask(__name__)

# Load character set
char_file = "characters.json"
if not os.path.exists(char_file):
    CHARACTERS = "-abcdefghijklmnopqrstuvwxyz"
    with open(char_file, "w") as f:
        json.dump(CHARACTERS, f)
else:
    with open(char_file, "r") as f:
        CHARACTERS = json.load(f)

char_to_idx = {c: i for i, c in enumerate(CHARACTERS)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Load model
model_file = "crnn_model.pt"
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file '{model_file}' not found.")

model = torch.jit.load(model_file)
model.eval()

# Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

# CTC Greedy Decoding
def ctc_decoder(output):
    output = output.softmax(2).argmax(2).squeeze(1).tolist()
    decoded_text = []
    prev_char = None
    
    for idx in output:
        if idx != prev_char and idx < len(CHARACTERS):
            decoded_text.append(CHARACTERS[idx])
        prev_char = idx
    
    return ''.join(decoded_text)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'Không có ảnh được tải lên'}), 400

    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        input_tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(input_tensor)
        predicted_text = ctc_decoder(output)
        
        return jsonify({'result': predicted_text})
    except Exception as e:
        return jsonify({'error': f'Lỗi xử lý ảnh: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
