from asyncore import read
import csv
import glob
import os
from os.path import join
from os import makedirs
import sys

import numpy as np
import cv2
import imageio

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
        self.fps = fps
        frame_counter = 0   
        self.imgs = []

        while True:
            # 1フレームずつ取得する。
            ret, frame = cap.read()
            if not ret:
                break
            frame_counter += 1

            if (frame_counter >= self.thresh): #フレームカウントがthreshを超えたら処理する
                # Note the input is expected to be RGB so do not convert again !
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                self.imgs.append(frame)
                frame_counter = 0 #フレームカウントを０に戻す
        self.imgs = np.stack(self.imgs, axis=0)

    def __len__(self):
        return len(self.imgs)

    def evaluate(self, results):
        eval_result = {
            "some fish": 1.0,
            "run_time": 1.0,
        }
        return eval_result

    def write_results(self, results):
        problem_name = self.result_dir.split("/")[-1]
        """
        Write the results into file with required csv format
        """
        with open(join(self.result_dir, f"output-{problem_name}.csv"), "w") as f:
            f.write(f"{problem_name},\n")
            f.write("5,\n")
            for i, elem in enumerate(results):
                f.write(f"{i+1}, {elem}\n")

    def export_images(self, export_dir):
        makedirs(export_dir, exist_ok=True)
        for i, img in enumerate(self.imgs):
            FILENAME = f"{np.ceil(i*self.fps).astype(np.int32):04d}.png"
            imageio.imwrite(join(export_dir, FILENAME), img)

    def get_images(self, index):
        return self.imgs[index]
    
    def get_near_frames(self, index, num_frames):
        pass

    def get_annotaions(self):
        pass
