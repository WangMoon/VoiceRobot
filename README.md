# 前期环境准备
```bash
# 创建环境
conda create -n demo python=3.10 -y
# 激活环境
conda activate demo
# 安装 torch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# 安装其他依赖
pip install transformers==4.38
pip install sentencepiece==0.1.99
pip install einops==0.8.0
pip install protobuf==5.27.2
pip install accelerate==0.33.0
pip install streamlit==1.37.0
pip install funasr==1.1.2
pip install modelscope==1.16.1

# 安装npm
#apt-get install nodejs npm
```

# FunASR准备
```python
from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
# 首次加载模型会需要下载
record_file = "/root/streamlit-robot/VoiceRecord/"
model = AutoModel(model="paraformer-zh",  
                    vad_model="fsmn-vad", 
                    punc_model="ct-punc", 
                  # spk_model="cam++" # spk_model: speaker model
                    log_level="error", # log_level: error, warning, info, debug
                    hub="ms" # hub：表示模型仓库，ms为选择modelscope下载，hf为选择huggingface下载。
                  )
res = model.generate(input=f"{record_file}/autumnrain.mp3", 
            batch_size_s=300, 
            )
text = res[0]['text']
print(text)
```

# ChatTTS

见文件 chattts_colab.ipynb
- 在项目文件夹下，下载ChatTTS repo
```bash
git clone https://github.com/2noise/ChatTTS.git
```

- 安装ChatTTS相关依赖文件
```bash
pip install -r /content/ChatTTS/requirements.txt
ldconfig /usr/lib64-nvidia
```
- 在项目文件夹下，下载ChatTTS相关模型文件
```python
import torch

torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")

from ChatTTS import ChatTTS
from ChatTTS.tools.logger import get_logger
from ChatTTS.tools.normalizer import normalizer_en_nemo_text, normalizer_zh_tn
from IPython.display import Audio

logger = get_logger("ChatTTS", format_root=True)
chat = ChatTTS.Chat(logger)

# try to load normalizer
try:
    chat.normalizer.register("en", normalizer_en_nemo_text())
except ValueError as e:
    logger.error(e)
except:
    logger.warning("Package nemo_text_processing not found!")
    logger.warning(
        "Run: conda install -c conda-forge pynini=2.1.5 && pip install nemo_text_processing",
    )
try:
    chat.normalizer.register("zh", normalizer_zh_tn())
except ValueError as e:
    logger.error(e)
except:
    logger.warning("Package WeTextProcessing not found!")
    logger.warning(
        "Run: conda install -c conda-forge pynini=2.1.5 && pip install WeTextProcessing",
    )

    # use force_redownload=True if the weights have been updated.
chat.load(source="huggingface") 
# 从huggingface下载可能会超时，需要下载的文件全部在asset文件夹中
# 保证asset文件夹原封不动在项目文件夹，ChatTTS可以正常使用

```


