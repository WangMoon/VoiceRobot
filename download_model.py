#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('iic/SenseVoiceSmall')
model_dir = snapshot_download('iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
model_dir = snapshot_download('iic/punc_ct-transformer_cn-en-common-vocab471067-large')
model_dir = snapshot_download('iic/punc_ct-transformer_cn-en-common-vocab471067-large',cache_dir='/root/VoiceRobot/iic')