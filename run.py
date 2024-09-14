from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from datetime import datetime
from scipy.io.wavfile import write
from test_funasr import load_model, asr
from test_ChatTTS import model_ChatTTS, wav_save

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'downloads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load the models once at the start
asr_model = load_model()
chat_model = model_ChatTTS()
chat_model.load()

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
        text = asr(asr_model, filepath)
        
        # Generate audio using ChatTTS
        texts = [text]
        wavs = chat_model.infer(texts)
        
        # Save the generated audio
        output_filename = datetime.now().strftime("%Y%m%d%H%M%S") + '.wav'
        output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
        write(output_filepath, 24000, wavs[0])
        
        return jsonify({'text': text, 'audio_url': f'/downloads/{output_filename}'})
    return jsonify({'error': 'Failed to upload audio'})

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    # 将host设置为'0.0.0.0'以允许外部访问
    app.run(host='0.0.0.0', port=5000, debug=True)