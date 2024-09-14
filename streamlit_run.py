import streamlit as st
import requests
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import numpy as np
import av
import io
import base64

# 服务器地址，替换为你的服务器 URL
SERVER_URL = "http://<your-server-ip>:5000"

# 页面设置
st.title("Audio Recorder and TTS Application")

# 自定义音频处理器，处理录音数据
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.audio_buffer.extend(audio.flatten())
        return frame

# 创建音频缓冲区
webrtc_ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

if webrtc_ctx.audio_processor:
    audio_data = np.array(webrtc_ctx.audio_processor.audio_buffer, dtype=np.float32)
    if st.button("Save Recording"):
        # 将音频数据保存为 WAV 文件
        audio_bytes = io.BytesIO()
        np.save(audio_bytes, audio_data)
        audio_bytes.seek(0)
        audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')

        # 上传音频数据到服务器
        response = requests.post(f"{SERVER_URL}/upload", json={"data": audio_base64})
        result = response.json()

        # 显示语音识别结果
        st.write("Transcription Result: ", result['text'])

        # 播放生成的音频
        audio_url = f"{SERVER_URL}/downloads/{result['audio_filename']}"
        st.audio(audio_url, format='audio/wav')