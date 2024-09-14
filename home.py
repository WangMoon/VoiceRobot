from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from datetime import datetime
from scipy.io.wavfile import write
from test_funasr import load_model, asr
from test_ChatTTS import model_ChatTTS, wav_save
import threading

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
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio_data' in request.files:
        audio_data = request.files['audio_data']
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + '.wav'
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        audio_data.save(filepath)
        
        # Perform ASR on the uploaded audio file
        text = asr(asr_model, filepath)
        
        # Start a new thread to handle TTS
        threading.Thread(target=process_tts, args=(text,)).start()
        
        return jsonify({'text': text})
    return jsonify({'error': 'Failed to upload audio'})

# def process_tts(text):
#     # Generate audio using ChatTTS
#     texts = [text]
#     wavs = chat_model.infer(texts)
    
#     # Save the generated audio
#     output_filename = datetime.now().strftime("%Y%m%d%H%M%S") + '.wav'
#     output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
#     write(output_filepath, 24000, wavs[0])
    
#     # Notify the front-end (this can be done via WebSocket or polling)
#     # For simplicity, we will use polling in this example

def process_tts(text):
    texts = [text]
    wavs = chat_model.infer(texts)
    
    # 使用识别的文本作为文件名的一部分保存音频
    safe_text = "".join([c for c in text if c.isalnum()])[:10]  # 简单过滤并限制长度
    output_filename = f"{safe_text}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
    write(output_filepath, 24000, wavs[0])


@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route('/check_audio/<text>')
def check_audio(text):
    safe_text = "".join([c for c in text if c.isalnum()])[:10]
    for filename in os.listdir(OUTPUT_FOLDER):
        if filename.startswith(safe_text):
            return jsonify({'audio_url': f'/downloads/{filename}'})
    return jsonify({'audio_url': None})


if __name__ == '__main__':
    # 将host设置为'0.0.0.0'以允许外部访问
    app.run(host='0.0.0.0', port=5000, debug=True)