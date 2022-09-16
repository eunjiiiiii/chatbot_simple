import argparse


class Configs():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--audio_path", default=r"./data/intro.wav", help="audio file sampled as 16k hz")
        self.parser.add_argument("--img_path", default=r"./data/paint.jpg", help="reference image")
        self.parser.add_argument("--save_path", default=r"./data", help="save path")
        self.parser.add_argument("--model_path", default=r"./model/audio2head.pth.tar", help="pretrained model path")
        self.parser.add_argument("--config_file", default=r"./model/vox-256.yaml", help="vox-256 yaml path")
        self.parser.add_argument("--parameters", default=r"./model/parameters.yaml", help="parameters yaml path")

    def parse(self):
        return self.parser.parse_args()
