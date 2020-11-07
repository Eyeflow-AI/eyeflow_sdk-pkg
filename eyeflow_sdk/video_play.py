"""
SiliconLife Eyeflow
Video manipulation functions

Author: Alex Sobral de Freitas
"""

import os
import json
import datetime
from bson.objectid import ObjectId
import cv2
#---------------------------------------------------------------------------

class VideoPlay():
    def __init__(self, filename, window_size): #, video_scale):
        self._window_size = window_size
        #self._frame_scale = video_scale
        self._video_cap = cv2.VideoCapture(filename)

        self._video_frames = {}
        self._min_frame = 0
        self._max_frame = 0
        self._video_open = True

        success, video_frame = self._video_cap.read()
        if not success:
            raise Exception("Fail to read video: {}".format(filename))

        self._video_frames[str(self._min_frame)] = video_frame

    def is_open(self):
        return self._video_open


    def get_total_frames(self):
        return int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))


    def get_frame_rate(self):
        return int(self._video_cap.get(cv2.CAP_PROP_FPS))


    def get_frame(self, frame):
        if frame < self._min_frame:
            return self._video_frames[str(self._min_frame)], self._min_frame
        elif frame >= self._max_frame:
            while self._max_frame < frame:
                if self._video_open:
                    success, video_frame = self._video_cap.read()
                    if not success:
                        self._video_open = False
                        return self._video_frames[str(self._max_frame)], self._max_frame
                    else:
                        self._max_frame += 1
                        self._video_frames[str(self._max_frame)] = video_frame

        self.discard_frames_window()
        return self._video_frames[str(frame)], frame


    def discard_frames_window(self):
        if (self._max_frame - self._min_frame) < self._window_size:
            return

        while (self._max_frame - self._min_frame) > self._window_size:
            del self._video_frames[str(self._min_frame)]
            self._min_frame += 1


    def extract_frames(self, num_frames, dest_path):
        f_max = self.get_total_frames()
        f_count = 0
        f_step = f_max // num_frames
        frame = f_step
        while f_count <= num_frames:
            img, frame = self.get_frame(frame)
            img_name = str(ObjectId())
            cv2.imwrite(os.path.join(dest_path, img_name + '.jpg'), img)
            img_data = {
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "img_height": img.shape[0],
                "img_width": img.shape[1],
                "detections": {"instances":[]},
                "annotations": {"instances":[]}
            }

            with open(os.path.join(dest_path, img_name + '_data.json'), 'w', newline='', encoding='utf8') as fp:
                json.dump(img_data, fp, ensure_ascii=False, indent=2)

            frame += f_step
            f_count += 1
#----------------------------------------------------------------------------------------------------------------------------------

def get_video_info(video_path, video_file):
    # log.info("Get video info {}".format(video_file))

    file = os.path.join(video_path, video_file)
    if not os.path.isfile(file):
        raise Exception('Video not found: ' + file)

    vid_cap = VideoPlay(file, 300)

    total_frames = vid_cap.get_total_frames()
    total_time = round(total_frames / vid_cap.get_frame_rate(), 2)

    video_frame, _ = vid_cap.get_frame(0)

    frame_height = video_frame.shape[0]
    frame_width = video_frame.shape[1]

    video_info = {
        "video_file": video_file,
        "total_frames": total_frames,
        "total_time": total_time,
        "frame_height": frame_height,
        "frame_width": frame_width
    }

    return json.dumps(video_info)
#----------------------------------------------------------------------------------------------------------------------------------
