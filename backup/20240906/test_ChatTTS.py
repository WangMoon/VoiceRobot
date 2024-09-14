import torch
import torchaudio
import os
from datetime import datetime
from scipy.io.wavfile import write
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")

from ChatTTS import ChatTTS
from ChatTTS.tools.logger import get_logger
from ChatTTS.tools.normalizer import normalizer_en_nemo_text, normalizer_zh_tn
from IPython.display import Audio

def model_ChatTTS():
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
        return chat
    
def wav_save(wav, output_path):
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + '.wav'
    filepath = os.path.join(output_path, filename)
    write(filepath, 24000, wav)
    return filepath

def main():
    # chat.load()
    chat.load()

    texts = [
            "我觉得像我们这些写程序的人，他，我觉得多多少少可能会对开源有一种情怀在吧我觉得开源是一个很好的形式。现在其实最先进的技术掌握在一些公司的手里的话，就他们并不会轻易的开放给所有的人用。"
    ] 

    wavs = chat.infer(texts)


    # 假设 wavs[0] 是一个 numpy 数组，包含音频数据
    # 24_000 是采样率
    output_path = "downloads"
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + '.wav'
    filepath = os.path.join(output_path, filename)
    write(filepath, 24000, wavs[0])

# Audio(wavs[0], rate=24_000, autoplay=True)

if __name__ == '__main__':
    main()