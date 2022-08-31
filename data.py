from asyncore import read
import csv
import glob
from os.path import join
from os import makedirs
import sys

import numpy as np
import cv2


class Dataset():
    def __init__(self, data_dir, result_dir, fps=5):
        makedirs(result_dir, exist_ok=True)
        self.video_path = glob.glob(join(data_dir, "*.MP4"))[0]
        self.anno_path = glob.glob(join(data_dir, "*.csv"))[0]
        self.result_dir = result_dir

        cap = cv2.VideoCapture(self.video_path)

        read_fps= cap.get(cv2.CAP_PROP_FPS) #cv2.VideoCaptureが動画1秒あたり何枚を読み込むのかを取得（これは自動で設定される）
        self.original_fps = read_fps
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.thresh = read_fps / fps #フレーム何枚につき1枚処理するか
        frame_counter = 0   
        self.imgs = []

        while True:
            # 1フレームずつ取得する。
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1

            if (frame_counter >= self.thresh): #フレームカウントがthreshを超えたら処理する
                self.imgs.append(frame)
                frame_counter = 0 #フレームカウントを０に戻す
        self.imgs = np.stack(self.imgs, axis=0)

    def evaluate(self, results):
        eval_result = {
            "some fish": 1.0,
            "run_time": 1.0,
        }
        return eval_result

    def write_results(self, results):
        """
        Write the results into file with required csv format
        """
        with open(join(self.result_dir, f"output.csv"), "w") as f:
            f.writelines(["Good Job !!"])


    def get_images(self, index):
        return self.imgs[index]
    
    def get_near_frames(self, index, num_frames):
        pass

    def get_annotaions(self):
        pass
