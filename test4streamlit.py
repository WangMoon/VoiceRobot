import os
import streamlit as st
import qianfan
import pyaudio
import requests
import json
import base64
import time

timer = time.perf_counter

API_KEY = ''
SECRET_KEY = ''

# 文件格式
AUDIO_FILE = './recorded_audio.pcm'  # 只支持 pcm/wav/amr 格式，极速版额外支持m4a 格式
FORMAT = AUDIO_FILE[-3:]  # 文件后缀只支持 pcm/wav/amr 格式，极速版额外支持m4a 格式
CUID = '123456PYTHON'
# 采样率
RATE = 16000  # 固定值
# 普通版
DEV_PID = 1537  # 1537 表示识别普通话，使用输入法模型。根据文档填写PID，选择语言及识别模型
ASR_URL = 'http://vop.baidu.com/server_api'
TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'
API_KEY2 = ""
SECRET_KEY2 = ""
# 设置环境变量
os.environ["QIANFAN_ACCESS_KEY"] = ""
os.environ["QIANFAN_SECRET_KEY"] = ""

# 初始化聊天补全对象
chat_comp = qianfan.ChatCompletion()
def generate_speech(text,perr):
    url = "https://tsn.baidu.com/text2audio"

    payload = 'tex='+text+'&tok=' + get_access_token() + '&cuid=f5qRhsC3CGjo52E5rM3fM8Ibd56Vlabj&ctp=1&lan=zh&spd=5&pit=5&vol=5&per='+str(perr)+'&aue=3'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': '*/*'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    audio_data = response.content
    return audio_data
    # st.audio(audio_data, format='audio/mp3')
def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url2 = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY2, "client_secret": SECRET_KEY2}
    return str(requests.post(url2, params=params).json().get("access_token"))

class Audio_AI:
    def __init__(self):
        # 配置录音参数
        self.FORMAT = pyaudio.paInt16  # 音频格式，可以根据需要选择
        self.CHANNELS = 1  # 声道数（单声道）
        self.RATE = 16000  # 采样率
    def get_token(self):
        # 组装数据
        params = {'grant_type': 'client_credentials', 'client_id': API_KEY, 'client_secret': SECRET_KEY}
        response = requests.post(TOKEN_URL, timeout=10, params=params)
        if response.status_code == 200:
            # 转成JSON
            result = json.loads(response.text)
            # print(result)
            if ('access_token' in result.keys() and 'scope' in result.keys()):
                return result['access_token']

    def asr_audio(self, speech_data, length, token):
        # 进行base64编码
        speech = base64.b64encode(speech_data)
        speech = str(speech, 'utf-8')
        print(speech)
        headers = {"Content-Type": "application/json"}
        # 封装识别API请求的参数
        body = {'dev_pid': DEV_PID,
                # "lm_id" : LM_ID,    #测试自训练平台开启此项
                'format': FORMAT,
                'rate': RATE,
                'token': token,
                'cuid': CUID,
                'channel': 1,
                'speech': speech,
                'len': length
                }
        result = {}
        begin = timer()
        json_data = json.dumps(body, indent=4)  # indent 参数可选，用于格式化输出
        response = requests.post(ASR_URL, headers=headers, timeout=10, data=json_data)
        print(response.status_code)
        if response.status_code == 200:
            # 转成JSON
            result_json = json.loads(response.text)
            print(result_json)
            result = result_json["result"]
        else:
            print("请求失败")
        print("Request time cost %f" % (timer() - begin))
        return result

    def run_audio_asr(self, speech_data):
        # 获取token
        token = self.get_token()
        print("get token:", token)
        length = len(speech_data)
        print('l',length)
        if length == 0:
            print("发送识别失败，录音数据可能为空")
        else:
            result = self.asr_audio(speech_data, length, token)
            print(result)
            return result

def click_button():
    st.session_state.button = not st.session_state.button

if 'button' not in st.session_state:
    st.session_state.button = False
    st.session_state.a = 0
    st.session_state.te = ""
def main_page():
    st.title("Chat with WENXIN")
    # 定义单选按钮的选项
    gender_options = ['小美', '小宇', '逍遥', '丫丫']

    # 使用 st.radio 创建单选按钮组
    selected_gender = st.sidebar.radio(
        label="Select your gender:",
        options=gender_options,
        index=1,  # 默认选中第一个选项（“Male”）
        key='gender_selection',
        help='Please choose your gender from the provided options.'
    )

    # 显示用户选择的结果
    st.write(f"You selected: {selected_gender}")
    selected_gender = gender_options.index(selected_gender)
    if selected_gender > 1:
        selected_gender += 1
    # 如果没有历史会话则初始化“你好，请问有什么问题吗？”
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["messages"].append({"role": "assistant", "content": "你好，请问有什么问题吗？"})
    # 有会话就输入会话
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == 'assistant':
                st.audio(generate_speech(msg["content"], selected_gender), format='audio/mp3')
    t = st.sidebar.button("语音输入/停止录音", key="1", on_click=click_button)
    t2 = st.sidebar.button("确定", key="2")
    if not st.session_state.button:
        st.sidebar.write("点击即使用")
    if st.session_state.button and st.session_state.a == 0:
        time.sleep(0.5)

    text = ""
    if st.session_state.button and st.session_state.a == 0:
        st.session_state.a = 1
        audio = pyaudio.PyAudio()
        # 打开音频输入流
        stream = audio.open(format=pyaudio.paInt16, channels=1,
                            rate=RATE, input=True,
                            frames_per_buffer=1024)
        st.write("正在录制...")
        st.sidebar.write("录音ing")
        print("开始录制...")
        frames = []
        # 录制音频
        while 1:
            data = stream.read(1024)
            frames.append(data)
            st.session_state.fr = frames
    if not st.session_state.button and st.session_state.a == 1:
        st.session_state.a = 0
        print("停止")
        # 打开音频输入流
        k = Audio_AI()
        speech_data = b''.join(st.session_state.fr)
        text = k.run_audio_asr(speech_data)[0]
        st.sidebar.write("录音结果为：", text)
        st.session_state.te = text
    st.session_state.w = 0
    if st.session_state.te != "":
        if t2:
            text = st.session_state.te
            st.session_state.w = 1
    # 使用chat_input函数获取用户的输入，并将其存储在变量prompt中。如果用户输入了内容，条件判断为真。
    if prompt := st.chat_input() or text != "" and st.session_state.w == 1:
        # 把输入添加到message中
        if text != "":
            prompt = text
        st.session_state.messages.append({"role": "user", "content": prompt})
        # 显示在界面中
        with st.chat_message("user"):
            st.markdown(prompt)
        # 使用chat_comp生成回复，丢弃第一句话（欢迎话）
        resp = chat_comp.do(model="ERNIE-Speed-8K", messages=st.session_state.messages[1:], stream=True)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for r in resp:
                chunk = r["body"]["result"]
                full_response += chunk
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.audio(generate_speech(full_response, selected_gender), format='audio/mp3')
            st.write("输出完毕！")
        msg = full_response
        # 把回复添加到message中
        st.session_state.messages.append({"role": "assistant", "content": msg})

if __name__ == '__main__':
    main_page()
