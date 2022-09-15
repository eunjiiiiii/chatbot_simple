import sys
import io
from glob import glob

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QLabel, QLineEdit, QHBoxLayout, QVBoxLayout, QPushButton, QTextBrowser, QMainWindow, QStyle, QSizePolicy, QTextEdit, QMessageBox, QSlider
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QMovie

from get_pie import graph_func
from model.TTS2 import synthesizer
from model import T2F
from emotionchat_engine import EmotionChat, final_emotion
import emotionchat_config as config

import torch
import random
import os
import shutil
import numpy as np
from pydub import AudioSegment
import soundfile as sf
import librosa
from audio_recoder import AudioRecord


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RecordWorker(QThread):
    record_fin_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(RecordWorker, self).__init__()
        self.main = parent
        self.recorder = AudioRecord(chunk=1024, channels=1, rate=22050, silence_second=2, silence_threshold=3000)
        self.recorder.open()
        self.FILE_USER_VOICE = "./resources/recorded.wav"

    @pyqtSlot()
    def setupRecord(self):
        self.recorder.get_audio()
        self.recorder.save_audio(self.FILE_USER_VOICE)


class MediaWorker(QThread):
    media_fin_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super(MediaWorker, self).__init__()
        self.main = parent
        # self.ANSWERS = {}
        # for k, v in config.ANSWER.items():
        #     for i, ans in enumerate(v):
        #         self.ANSWERS[ans] = f"{k}_{i}.wav"
        self.ANSWERS = {''.join(v): "image_" + k + "_0.wav" for k, v in config.ANSWER.items()}

        self.synthesizer = synthesizer.Synthesizer()
        self.synthesizer.load("./model/TTS2/ckpt", num_speakers=3,
                              checkpoint_step=None, inference_prenet_dropout=False)
        self.t2f = T2F.text2face.Text2Face('./model/T2F/model/audio2head.pth.tar', src_dir='./resources')

        self.FILE_BOT_VOICE = "./resources/audios/synthesized.wav"

    @pyqtSlot()
    def setupMedia(self):

        key = self.main.result_dict["answer"]
        key = ''.join(key)

        if key in self.ANSWERS.keys():
            key = self.ANSWERS[key]
            key = key.replace(".wav", ".mp4")
            self.media_fin_signal.emit(key)
        else:
            answer_audios = []
            for idx, answer in enumerate(self.main.result_dict["answer"]):
                # Generate Bot Voice
                self.synthesizer.synthesize(texts=answer,
                                            base_path=self.FILE_BOT_VOICE.replace(".wav", ".png"),
                                            speaker_ids=[1],
                                            attention_trim=False,
                                            base_alignment_path=None,
                                            isKorean=True)
                pydub_out = AudioSegment.from_wav(self.FILE_BOT_VOICE)
                last_2_seconds = pydub_out[-10:]
                do_it_over = last_2_seconds * 40
                with_style = pydub_out.append(do_it_over)
                with_style.fade_out(1000)
                export_name = self.FILE_BOT_VOICE.replace("synthesized", f"synthesized_{idx}")
                with_style.export(export_name, format='wav')

                answer_audios.append(export_name)

            converged = []
            for a in answer_audios:
                y, sr = librosa.load(a)
                mute_add = np.append(np.zeros(11025), y)
                converged = np.append(converged, mute_add)
            sf.write("./resources/synthesized.wav", converged, 22050, "PCM_16")

            self.t2f(os.path.basename(self.FILE_BOT_VOICE), 'image.jpg', self.main.turn_cnts["turn_cnt"])
            filename = f"image_synthesized_{self.main.turn_cnts['turn_cnt']}.mp4"
            shutil.move(os.path.join("./resources", filename),
                        os.path.join(self.main.DIR_VIDEOS, filename))
            self.media_fin_signal.emit(filename)


class MyApp(QMainWindow):
    record_signal = pyqtSignal()
    media_signal = pyqtSignal()

    def __init__(self):
        super(MyApp, self).__init__()
        self.wid = QWidget(self)

        # Google Speech To Text API
        credential_path = "./google-stt-api-key/axiomatic-spark-357606-54b9f8b39365.json"
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
        self.client = speech.SpeechClient()

        # Video & Audio Interface
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        self.audioPlayer = QMediaPlayer()
        self.audioPlayButton = QPushButton()
        self.audioPlayButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.audioPlayButton.clicked.connect(self.playNarrAudio)
        self.audioPositionSlider = QSlider(Qt.Horizontal)
        self.audioPositionSlider.setRange(0, 0)
        self.audioPositionSlider.sliderMoved.connect(self.audioSetPosition)

        self.audioPlayer.stateChanged.connect(self.audioStateChanged)
        self.audioPlayer.positionChanged.connect(self.audioPositionChanged)
        self.audioPlayer.durationChanged.connect(self.audioDurationChanged)

        self.record_th = RecordWorker(parent=self)
        self.record_th.record_fin_signal.connect(self.record_audio)
        self.record_signal.connect(self.record_th.setupRecord)
        self.record_th.start()

        # User Inputs Resources
        # self.FILE_USER_VOICE = "./resources/recorded.wav"
        self.FILE_SENTIMENT_PIE_GIF = "./resources/simWave.gif"
        self.DIR_VIDEOS = "./resources/videos"
        # self.FILE_BOT_VOICE = "./resources/audios/synthesized.wav"
        # self.ANSWERS = {}
        # for k, v in config.ANSWER.items():
        #     for i, ans in enumerate(v):
        #         self.ANSWERS[ans] = f"{k}_{i}.wav"

        # Text To Speech
        # self.synthesizer = synthesizer.Synthesizer()
        # self.synthesizer.load("./model/TTS2/ckpt", num_speakers=3,
        #                       checkpoint_step=None, inference_prenet_dropout=False)

        # Text To Face
        # self.t2f = T2F.text2face.Text2Face('./model/T2F/model/audio2head.pth.tar', src_dir='./resources')
        self.media_th = MediaWorker(parent=self)
        self.media_th.media_fin_signal.connect(self.playMedia)
        self.media_signal.connect(self.media_th.setupMedia)
        self.media_th.start()

        # Sentiment Pie chart Initialization
        self.pie_label = QLabel()
        self.before_sentiments = [1., 1., 1., 1., 1., 1.]
        self.pie_chart = None
        self.setPIE(self.before_sentiments)
        self.pie_label.setMovie(self.pie_chart)
        self.playPIE()

        # Indicators
        self.intent_indicator = QTextEdit()
        self.intent_turn_indicator = QTextEdit()
        self.turn_indicator = QTextEdit()

        # Chat Interface
        self.tb = QTextBrowser()

        # Chat Input Layout
        self.qle = QLineEdit(self)

        # Mic on Button
        # self.btn = QPushButton("&s말하기 ", self)
        self.btn = QPushButton("말하기 ", self)

        self.initUI()

        seed_everything(1234)
        self.emotion_chat = EmotionChat()
        self.chatbot_engine_init()

    def initUI(self):
        self.pie_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pie_label.setAlignment(Qt.AlignCenter)

        # Interface attributes
        self.intent_indicator.setStyleSheet("font-size:20px;font-weight: bold;")
        self.intent_indicator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # intent_font = QFontMetrics(self.intent_indicator.font())
        # intent_height = intent_font.height() + (1 + self.intent_indicator.frameWidth()) * 2
        # self.intent_indicator.setFixedHeight(intent_height)
        self.intent_indicator.setReadOnly(True)
        self.intent_turn_indicator.setStyleSheet("font-size:20px;font-weight: bold;")
        self.intent_turn_indicator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.intent_turn_indicator.setReadOnly(True)
        self.turn_indicator.setStyleSheet("font-size:20px;font-weight:bold;")
        self.turn_indicator.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.turn_indicator.setReadOnly(True)
        self.tb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tb.setStyleSheet("font-size:20px;font-weight:bold;")
        self.tb.setReadOnly(True)

        self.qle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn.clicked.connect(self.record_audio)

        hbox = QHBoxLayout()
        chat_input_layout = QHBoxLayout()
        vbox_left = QVBoxLayout()
        vbox_right = QVBoxLayout()
        indicator_layout = QHBoxLayout()

        # left side
        lbl1 = QLabel("감정")
        lbl1.setAlignment(Qt.AlignCenter)
        fnt = lbl1.font()
        fnt.setPointSize(20)
        lbl1.setFont(fnt)
        vbox_left.addWidget(lbl1)
        vbox_left.addWidget(self.pie_label)
        layout1 = QVBoxLayout()
        layout1.addWidget(QLabel("인텐트"))
        layout1.addWidget(self.intent_indicator)
        indicator_layout.addLayout(layout1)

        layout2 = QVBoxLayout()
        layout2.addWidget(QLabel("인텐트 대화 제한"))
        layout2.addWidget(self.intent_turn_indicator)
        indicator_layout.addLayout(layout2)

        layout3 = QVBoxLayout()
        layout3.addWidget(QLabel("총 대화 수"))
        layout3.addWidget(self.turn_indicator)
        indicator_layout.addLayout(layout3)
        vbox_left.addLayout(indicator_layout, stretch=1)

        vbox_left.addWidget(self.tb, stretch=20)
        chat_input_layout.addWidget(self.qle)
        chat_input_layout.addWidget(self.btn)
        vbox_left.addLayout(chat_input_layout, stretch=1)

        # right side
        videoWidget = QVideoWidget()
        videoWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vbox_right.addWidget(videoWidget)
        lbl2 = QLabel("마음결")
        lbl2.setAlignment(Qt.AlignCenter)
        lbl2.setFont(fnt)
        vbox_right.addWidget(lbl2)
        self.mediaPlayer.setVideoOutput(videoWidget)

        hbox.addLayout(vbox_left, stretch=1)
        hbox.addLayout(vbox_right, stretch=1)

        self.wid.setLayout(hbox)
        self.setCentralWidget(self.wid)
        self.setWindowTitle('IAI ChatBot Demo')
        self.resize(1100, 600)
        self.center()
        self.showMaximized()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def record_audio(self):
        # self.recorder.get_audio()
        # self.recorder.save_audio(self.FILE_USER_VOICE)
        self.record_signal.emit()
        self.sendMSG()

    def sendMSG(self):
        self.intent_indicator.clear()
        self.intent_turn_indicator.clear()
        self.turn_indicator.clear()
        self.btn.setEnabled(False)

        # Speech to Text
        wav_file = self.record_th.FILE_USER_VOICE
        # with io.open(wav_file, 'rb') as audio_file:
        #     audio_f = audio_file.read()
        # audio = types.RecognitionAudio(content=audio_f)
        # STT_config = types.RecognitionConfig(
        #     encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        #     sample_rate_hertz=22050,
        #     language_code='ko-KR')
        # response = self.client.recognize(STT_config, audio)
        # STT_result = ''
        # for result in response.results:
        #     STT_result += result.alternatives[0].transcript
        STT_result = self.qle.text()
        user_text = "User: " + STT_result + "\n"

        # Chat append
        cursor = self.tb.textCursor()
        block_format = cursor.blockFormat()
        block_format.setAlignment(Qt.AlignRight)
        cursor.mergeBlockFormat(block_format)
        self.tb.setTextCursor(cursor)
        self.tb.setTextColor(Qt.blue)
        self.tb.append(user_text)

        # text = STT_result.strip()
        # text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', re.sub('\(\d+일\)', '', text.strip())).replace('\n', ' ')

        # chat_inp = self.preliminaries.copy()
        # chat_inp.update({"text": STT_result, "wav_file": wav_file})
        self.result_dict = self.emotion_chat.run(text=STT_result, wav_file=wav_file,
                                                 pre_result_dict=self.result_dict, turn_cnts=self.turn_cnts)
        self.chat_post_process(self.result_dict)

        bot_text = "Bot: " + '\n'.join(self.result_dict["answer"]) + "\n"
        block_format.setAlignment(Qt.AlignLeft)
        cursor.mergeBlockFormat(block_format)
        self.tb.setTextCursor(cursor)
        self.tb.setTextColor(Qt.red)
        self.tb.append(bot_text)

        # indicator change
        intent_cursor = self.intent_indicator.textCursor()
        intent_format = intent_cursor.blockFormat()
        intent_format.setAlignment(Qt.AlignCenter)
        intent_cursor.mergeBlockFormat(intent_format)
        self.intent_indicator.setTextCursor(intent_cursor)
        self.intent_indicator.append(self.result_dict['intent'])

        intent_turn_cursor = self.intent_turn_indicator.textCursor()
        intent_turn_format = intent_turn_cursor.blockFormat()
        intent_turn_format.setAlignment(Qt.AlignCenter)
        intent_turn_cursor.mergeBlockFormat(intent_turn_format)
        self.intent_turn_indicator.setTextCursor(intent_turn_cursor)
        self.intent_turn_indicator.append(f"{self.result_dict['intent_turn_cnt'] + 1} / 2")

        turn_cursor = self.turn_indicator.textCursor()
        turn_format = turn_cursor.blockFormat()
        turn_format.setAlignment(Qt.AlignCenter)
        turn_cursor.mergeBlockFormat(turn_format)
        self.turn_indicator.setTextCursor(turn_cursor)
        self.turn_indicator.append(f"{self.turn_cnts['turn_cnt']}")

        self.qle.clear()

        self.btn.setEnabled(True)

        # self.playMEDIA()
        self.setPIE(self.result_dict['emotion_probs'])
        self.playPIE()
        self.media_signal.emit()

        if self.result_dict["answer"] == config.ANSWER["default_contents"]:
            self.playNarr()
        elif self.result_dict["current_phase"] == "/end_phase":
            self.exitPopup()

    @pyqtSlot(str)
    def playMedia(self, key):
        # self.playVIDEO(f"image_synthesized_{self.turn_cnt}")
        self.playVIDEO(key)

    def chat_post_process(self, result_dict):

        self.conversation_history.append(result_dict)
        self.turn_cnt += 1
        self.turn_cnts = {'turn_cnt': self.turn_cnt, 'intent_turn_cnt': self.intent_turn_cnt}

        if result_dict['current_phase'] == '/end_phase':
            self.chatbot_engine_exit(result_dict)

    def chatbot_engine_init(self):
        self.turn_cnt, self.intent_turn_cnt = 0, 0
        self.turn_cnts = {'turn_cnt': self.turn_cnt, 'intent_turn_cnt': self.intent_turn_cnt}
        self.result_dict = {
            "input": [], "intent": '', "entity": "", "state": "",
            "emotion": "", "emotions": [], "emotion_prob": [],
            "topics": [], "topic_prob": [],
            "answer": "", "previous_phase": "", "current_phase": "", "next_phase": [],
            "intent_turn_cnt": self.intent_turn_cnt
        }
        self.conversation_history = []
        self.final_emo_topic = None

    def chatbot_engine_exit(self, result_dict):
        self.final_emo_topic = final_emotion(result_dict)
        item_list = self.final_emo_topic.items()

        print('\n' + '-'*100)
        print('-' * 100)
        print('<최종 딕셔너리에서 최대 감정, 주제와 그에 대응하는 확률 출력>')
        print(item_list)
        print('-' * 100)
        print("-" * 100 + "\n\n")

    def setPIE(self, after):
        graph_func(self.FILE_SENTIMENT_PIE_GIF, self.before_sentiments, after)
        self.before_sentiments = after
        self.pie_chart = QMovie(self.FILE_SENTIMENT_PIE_GIF)
        self.pie_label.setMovie(self.pie_chart)

    def playPIE(self):
        self.pie_chart = QMovie(self.FILE_SENTIMENT_PIE_GIF)
        self.pie_label.setMovie(self.pie_chart)
        self.pie_chart.start()

    def playVIDEO(self, video_name):
        self.mediaPlayer.setMedia(QMediaContent(
            QUrl.fromLocalFile(os.path.join(self.DIR_VIDEOS, video_name))))
        self.mediaPlayer.play()

    def playNarr(self):
        result = list(self.final_emo_topic.keys())
        content = QMediaContent(QUrl.fromLocalFile(f"resources/narr/{result[0]}-{result[1]}.wav"))

        msg = QMessageBox()
        self.audioPlayer.setMedia(content)

        popup_control_layout = msg.layout()
        popup_control_layout.addWidget(self.audioPlayButton)
        popup_control_layout.addWidget(self.audioPositionSlider)

        msg.setWindowTitle("나레이션")
        msg.setText(f"인식된 감정: {result[0]}, 주제: {result[1]}\n프로그램을 종료하려면 OK\n다시 대화하려면 Cancel")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.buttonClicked.connect(self.system_exit)

        msg.exec_()

    def exitPopup(self):
        msg = QMessageBox()

        msg.setWindowTitle("종료")
        msg.setText("채팅이 종료되었습니다.\n프로그램을 종료하려면 OK\n다시 대화하려면 Cancel")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.buttonClicked.connect(self.system_exit)

        msg.exec_()

    def system_exit(self, i):
        if i.text() == "OK":
            self.close()
        else:
            self.chatbot_engine_init()
            self.intent_indicator.clear()
            self.intent_turn_indicator.clear()
            self.turn_indicator.clear()
            self.tb.clear()
            self.audioPlayer.stop()
            self.audioPlayButton = QPushButton()
            self.audioPlayButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.audioPlayButton.clicked.connect(self.playNarrAudio)
            self.audioPositionSlider = QSlider(Qt.Horizontal)
            self.audioPositionSlider.setRange(0, 0)
            self.audioPositionSlider.sliderMoved.connect(self.audioSetPosition)
            self.setPIE([1., 1., 1., 1., 1., 1.])
            self.playPIE()

    def playNarrAudio(self):
        if self.audioPlayer.state() == QMediaPlayer.PlayingState:
            self.audioPlayer.pause()
        else:
            self.audioPlayer.play()

    def audioStateChanged(self):
        if self.audioPlayer.state() == QMediaPlayer.PlayingState:
            self.audioPlayButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.audioPlayButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def audioSetPosition(self, position):
        self.audioPlayer.setPosition(position)

    def audioPositionChanged(self, position):
        self.audioPositionSlider.setValue(position)

    def audioDurationChanged(self, duration):
        self.audioPositionSlider.setRange(0, duration)


if __name__ == '__main__':
    for file in glob("./resources/videos/image_synthesized_*"):
        os.remove(file)
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())
