import os
import cv2
import yaml
import warnings
import argparse
import subprocess
import numpy as np
from skimage import io, img_as_float32
from scipy.io import wavfile
from scipy.interpolate import interp1d
import imageio
import python_speech_features
import pyworld

import torch

from model.T2F.modules.audio2pose import audio2poseLSTM
from model.T2F.modules.keypoint_detector import KPDetector
from model.T2F.modules.generator import OcclusionAwareGenerator
from model.T2F.modules.audio2kp import AudioModel3D


class Text2Face:
    def __init__(self, model_path, src_dir="model/T2F/data"):
        warnings.filterwarnings('ignore')
        model_basedir = os.path.dirname(model_path)
        self.src_dir = src_dir

        # init audio2pose
        self.audio2pose = audio2poseLSTM().cuda()
        ckpt = torch.load(model_path)

        self.audio2pose.load_state_dict(ckpt["audio2pose"])
        self.audio2pose.eval()

        # init kp_detector
        with open(os.path.join(model_basedir, "vox-256.yaml")) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                      **config['model_params']['common_params'])
        self.kp_detector = self.kp_detector.cuda()
        self.kp_detector.load_state_dict(ckpt["kp_detector"])
        self.kp_detector.eval()

        # init generator
        self.generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])
        self.generator = self.generator.cuda()
        self.generator.load_state_dict(ckpt["generator"])
        self.generator.eval()

        # init audio2kp
        self.audio2kp_opt = argparse.Namespace(**yaml.load(open(os.path.join(model_basedir, "parameters.yaml")),
                                             Loader=yaml.FullLoader))
        self.audio2kp = AudioModel3D(self.audio2kp_opt).cuda()
        self.audio2kp.load_state_dict(ckpt["audio2kp"])
        self.audio2kp.eval()

    def __call__(self, inp_wav, src_jpg, turn_cnt):
        os.makedirs(self.src_dir, exist_ok=True)
        TEMP_AUDIO = os.path.join(self.src_dir, "temp.wav")
        inp_wav = os.path.join(self.src_dir, inp_wav)
        src_jpg = os.path.join(self.src_dir, src_jpg)
        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (inp_wav, TEMP_AUDIO))
        output = subprocess.call(command, shell=True, stdout=None)

        audio_feat = self.get_audio_feature_from_audio(TEMP_AUDIO)
        frames = len(audio_feat) // 4

        img = io.imread(src_jpg)[:, :, :3]
        img = cv2.resize(img, (256, 256))
        img = np.array(img_as_float32(img))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).cuda()

        ref_pose_rot, ref_pose_trans = self.get_pose_from_audio(img, audio_feat)
        torch.cuda.empty_cache()

        audio_f = []
        poses = []
        pad = np.zeros((4, 41), dtype=np.float32)
        for i in range(0, frames, self.audio2kp_opt.seq_len // 2):
            temp_audio = []
            temp_pos = []
            for j in range(self.audio2kp_opt.seq_len):
                if i + j < frames:
                    temp_audio.append(audio_feat[(i + j) * 4: (i + j) * 4 + 4])
                    trans = ref_pose_trans[i + j]
                    rot = ref_pose_rot[i + j]
                else:
                    temp_audio.append(pad)
                    trans = ref_pose_trans[-1]
                    rot = ref_pose_rot[-1]

                pose = np.zeros([256, 256])
                self.draw_annotation_box(pose, np.array(rot), np.array(trans))
                temp_pos.append(pose)
            audio_f.append(temp_audio)
            poses.append(temp_pos)

        audio_f = torch.from_numpy(np.array(audio_f, dtype=np.float32)).unsqueeze(0)
        poses = torch.from_numpy(np.array(poses, dtype=np.float32)).unsqueeze(0)

        bs = audio_f.shape[1]
        predictions_gen = []
        total_frames = 0
        for bs_idx in range(bs):
            t = {}
            t["audio"] = audio_f[:, bs_idx].cuda()
            t["pose"] = poses[:, bs_idx].cuda()
            t["id_img"] = img

            kp_gen_source = self.kp_detector(img)

            gen_kp = self.audio2kp(t)
            if bs_idx == 0:
                startid = 0
                end_id = self.audio2kp_opt.seq_len // 4 * 3
            else:
                startid = self.audio2kp_opt.seq_len // 4
                end_id = self.audio2kp_opt.seq_len // 4 * 3

            for frame_bs_idx in range(startid, end_id):
                tt = {}
                tt["value"] = gen_kp["value"][:, frame_bs_idx]
                if self.audio2kp_opt.estimate_jacobian:
                    tt["jacobian"] = gen_kp["jacobian"][:, frame_bs_idx]

                out_gen = self.generator(img, kp_source=kp_gen_source, kp_driving=tt)

                predictions_gen.append(
                    (np.transpose(out_gen['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))

                total_frames += 1
                if total_frames >= frames:
                    break
            if total_frames >= frames:
                break

        # add black Frame
        predictions_gen.append(
            (np.transpose(torch.zeros(1, 3, 256, 256).data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))
        if not os.path.exists(os.path.join(self.src_dir, "temp")):
            os.makedirs(os.path.join(self.src_dir, "temp"))
        out_mp4_base = os.path.basename(src_jpg)[:-4] + "_" + os.path.basename(inp_wav)[:-4] + "_" + str(turn_cnt) + ".mp4"
        tmp_out_mp4 = os.path.join(self.src_dir, "temp", out_mp4_base)

        imageio.mimsave(tmp_out_mp4, predictions_gen, fps=25.0)

        out_mp4 = os.path.join(self.src_dir, out_mp4_base)
        cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (tmp_out_mp4, inp_wav, out_mp4)
        os.system(cmd)
        os.remove(tmp_out_mp4)

    def get_audio_feature_from_audio(self, audio_path, norm=True):
        sample_rate, audio = wavfile.read(audio_path)

        if len(audio.shape) == 2:
            if np.min(audio[:, 0]) <= 0:
                audio = audio[:, 1]
            else:
                audio = audio[:, 0]

        if norm:
            audio = audio - np.mean(audio)
            audio = audio / np.max(np.abs(audio))
            # mfcc 구하기
            a = python_speech_features.mfcc(audio, sample_rate)
            b = python_speech_features.logfbank(audio, sample_rate)
            c, _ = pyworld.harvest(audio, sample_rate, frame_period=10)
            c_flag = (c == 0.0) ^ 1
            c = self.inter_pitch(c, c_flag)
            c = np.expand_dims(c, axis=1)
            c_flag = np.expand_dims(c_flag, axis=1)
            frame_num = np.min([a.shape[0], b.shape[0], c.shape[0]])

            cat = np.concatenate([a[:frame_num], b[:frame_num], c[:frame_num], c_flag[:frame_num]], axis=1)
            return cat

    @staticmethod
    def inter_pitch(y, y_flag):
        frame_num = y.shape[0]
        i = 0
        last = -1
        while (i < frame_num):
            if y_flag[i] == 0:
                while True:
                    if y_flag[i] == 0:
                        if i == frame_num - 1:
                            if last != -1:
                                y[last + 1:] = y[last]
                            i += 1
                            break
                        i += 1
                    else:
                        break
                if i >= frame_num:
                    break
                elif last == -1:
                    y[:i] = y[i]
                else:
                    inter_num = i - last + 1
                    fy = np.array([y[last], y[i]])
                    fx = np.linspace(0, 1, num=2)
                    f = interp1d(fx, fy)
                    fx_new = np.linspace(0, 1, inter_num)
                    fy_new = f(fx_new)
                    y[last + 1:i] = fy_new[1:-1]
                    last = i
                    i += 1

            else:
                last = i
                i += 1
        return y

    def get_pose_from_audio(self, img, audio):
        num_frame = len(audio) // 4
        minv = np.array([-0.639, -0.501, -0.47, -102.6, -32.5, 184.6], dtype=np.float32)
        maxv = np.array([0.411, 0.547, 0.433, 159.1, 116.5, 376.5], dtype=np.float32)

        # generator = audio2poseLSTM().cuda()
        #
        # ckpt_para = torch.load(model_path)
        #
        # generator.load_state_dict(ckpt_para["audio2pose"])
        # generator.eval()

        audio_seq = []
        for i in range(num_frame):
            audio_seq.append(audio[i * 4:i * 4 + 4])

        audio = torch.from_numpy(np.array(audio_seq, dtype=np.float32)).unsqueeze(0).cuda()

        x = {}
        x["img"] = img
        x["audio"] = audio
        poses = self.audio2pose(x)

        # print(poses.shape)
        poses = poses.cpu().data.numpy()[0]

        poses = (poses + 1) / 2 * (maxv - minv) + minv
        rot, trans = poses[:, :3].copy(), poses[:, 3:].copy()
        return rot, trans

    @staticmethod
    def draw_annotation_box(image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""

        camera_matrix = np.array(
            [[233.333, 0, 128],
             [0, 233.333, 128],
             [0, 0, 1]], dtype="double")

        dist_coeefs = np.zeros((4, 1))

        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          camera_matrix,
                                          dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)


# if __name__ == '__main__':
#     t2f = Text2Face("./model/audio2head.pth.tar", src_dir="./data")
#     t2f("output.wav", "image.jpg")
