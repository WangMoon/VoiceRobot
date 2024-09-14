#基于flask编写网页，实现录音并保存音频功能
from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from test_funasr import load_model, asr

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_data' in request.files:
        audio_data = request.files['audio_data']
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + '.wav'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        audio_data.save(filepath)
        
        # Perform ASR on the uploaded audio file
        text = asr(model, filepath)
        
        return jsonify({'text': text})
    return jsonify({'error': 'Failed to upload audio'})

if __name__ == '__main__':
    app.run(debug=True)