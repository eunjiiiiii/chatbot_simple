from model.TTS2.synthesizer import Synthesizer
from model.T2F.text2face import Text2Face
from emotionchat_config import ANSWER
import soundfile as sf
import numpy as np
import librosa
from glob import glob
import os
import shutil


synthesizer = Synthesizer()
synthesizer.load("./model/TTS2/ckpt", num_speakers=3,
                 checkpoint_step=None, inference_prenet_dropout=False)
zeros = np.zeros(11025)

for k, v in ANSWER.items():
    for i, text in enumerate(v):
        synthesizer.synthesize(texts=text,
                               base_path=f"resources/audios/{k}_{i}.png",
                               speaker_ids=[1],
                               attention_trim=False,
                               base_alignment_path=None,
                               isKorean=True)
    file_list = glob(f"./resources/audios/{k}_*[0-9].wav")
    file_list = sorted(file_list)
    converged = []
    for file in file_list:
        y, sr = librosa.load(file)
        mute_add = np.append(zeros, y)
        converged = np.append(converged, mute_add)
    sf.write(f"./resources/audios/{k}.wav", converged, 22050, "PCM_16")
    for file in glob(f"./resources/audios/{k}_*[0-9].*"):
        os.remove(file)

t2f = Text2Face('./model/T2F/model/audio2head.pth.tar', src_dir='./resources/audios')

for k, v in ANSWER.items():
    t2f(f"{k}.wav", "../image.jpg", turn_cnt=0)
    filename = f"image_{k}_0.mp4"
    shutil.move(os.path.join("./resources/audios", filename),
                os.path.join("./resources/videos", filename))
