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
    # chat.load()
    return chat
    
def wav_save(wav, output_path):
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + '.wav'
    filepath = os.path.join(output_path, filename)
    write(filepath, 24000, wav)
    return filepath

def main():
    # chat.load()
    chat = model_ChatTTS()
    chat.load()

    texts = [
            "秋天的雨，是一把钥匙。它带着清凉和温柔，轻轻地，轻轻地，趁你没留意，把秋天的大门打开了。秋天的雨，有一盒五彩缤纷的颜料。你看，它把黄色给了银杏树，黄黄的叶子像一把把小扇子，扇哪扇哪，扇走了夏天的炎热。它把红色给了枫树，红红的枫叶像一枚枚邮票，飘哇飘哇，邮来了秋天的凉爽。金黄色是给田野的，看，田野像金色的海洋。橙红色是给果树的，橘子、柿子你挤我碰，争着要人们去摘呢！菊花仙子得到的颜色就更多了，紫红的、淡黄的、雪白的……美丽的菊花在秋雨里频频点头。秋天的雨，藏着非常好闻的气味。梨香香的，菠萝甜甜的，还有苹果、橘子，好多好多香甜的气味，都躲在小雨滴里呢！小朋友的脚，常被那香味勾住。"

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