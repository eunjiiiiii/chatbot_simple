import pyaudio
import wave
from array import array
import threading


class AudioRecord:
    def __init__(self, chunk, channels, rate, silence_second, silence_threshold, formats=pyaudio.paInt16):
        self.chunk = chunk
        self.format = formats
        self.channel = channels
        self.rate = rate
        self.silence_second = silence_second
        self.silence_threshold = silence_threshold

        self.keep_recording = False

        self.frames = []
        self.stream = None
        self.p = pyaudio.PyAudio()

    def thread_start(self):
        self.th = threading.Thread(target=self.recoding)
        self.th.start()

    def open(self):
        self.keep_recording = True
        self.stream = self.p.open(format=self.format,
                                  channels=self.channel,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)
        # self.thread_start()

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def save_audio(self, output_filename):
        wf = wave.open(output_filename, "wb")
        wf.setnchannels(self.channel)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.frames = []

    def get_audio(self):
        silence_count = 0
        while 1:
            data = self.stream.read(self.chunk)
            vol = max(array('h', data))
            self.frames.append(data)

            print(vol)

            if vol < self.silence_threshold:
                silence_count += 1
                if (int(self.rate / self.chunk * self.silence_second) + 1) <= silence_count:
                    break
            else:
                silence_count = 0

    def recoding(self):
        data = self.stream.read(self.chunk)
        self.frames.append(data)
        if self.keep_recording:
            self.thread_start()
        else:
            self.stream.close()
            self.p.terminate()


if __name__ == "__main__":
    CHUNK = 1024
    CHANNELS = 1
    RATE = 44100
    SILENCE_SECONDS = 2
    SILENCE_THRESHOLD = 1500
    WAVE_OUTPUT_FILENAME = "test.wav"

    record = AudioRecord(CHUNK, CHANNELS, RATE, SILENCE_SECONDS, SILENCE_THRESHOLD)
    record.open()
    for _ in range(5):
        record.get_audio()
        record.save_audio(f"test{_}.wav")
    record.close()
