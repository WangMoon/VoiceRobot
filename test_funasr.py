from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
def load_model():
    model = AutoModel(model="paraformer-zh",  
                    vad_model="fsmn-vad", 
                    punc_model="ct-punc", 
                  # spk_model="cam++" # spk_model: speaker model
                    log_level="error", # log_level: error, warning, info, debug
                    hub="ms" # hub：表示模型仓库，ms为选择modelscope下载，hf为选择huggingface下载。
                  )
    return model

def asr(model, record_file):
    res = model.generate(input=f"{record_file}", 
            batch_size_s=300, 
            )
    text = res[0]['text']
    return text


def main():
    
    model = load_model()
    while True:
        record_file = input("Please input the record file path: ")
        if record_file == "exit":
            break
        text = asr(model, record_file)
        print(text)

if __name__ == '__main__':
    main()
# record_file = "/root/streamlit-robot/VoiceRecord/"
# model = AutoModel(model="paraformer-zh",  
#                     vad_model="fsmn-vad", 
#                     punc_model="ct-punc", 
#                   # spk_model="cam++" # spk_model: speaker model
#                     log_level="error", # log_level: error, warning, info, debug
#                     hub="ms" # hub：表示模型仓库，ms为选择modelscope下载，hf为选择huggingface下载。
#                   )
# res = model.generate(input=f"{record_file}/autumnrain.mp3", 
#             batch_size_s=300, 
#             )
# text = res[0]['text']
# print(text)