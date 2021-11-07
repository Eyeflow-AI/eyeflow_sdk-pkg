"""
SiliconLife Eyeflow
Class for log batch of extracted images from detection

Author: Alex Sobral de Freitas
"""

import os

import json
import datetime
import pytz
import random
import cv2
from bson import ObjectId

from eyeflow_sdk.file_access import FileAccess
import eyeflow_sdk.img_utils as img_utils
from eyeflow_sdk.log_obj import log
#----------------------------------------------------------------------------------------------------------------------------------

MAX_EXTRACT_FILES = 800
THUMB_SIZE = 128

def clear_log(extract_path, max_files=MAX_EXTRACT_FILES):
    files_list = os.listdir(extract_path)
    if len(files_list) > max_files:
        date_list = [(filename, datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(extract_path, filename)))) for filename in files_list]
        exclude_list = sorted(date_list, key=lambda x: x[1])[:len(files_list) - max_files]
        for filename, _ in exclude_list:
            try:
                os.remove(os.path.join(extract_path, filename))
            except:
                pass
#----------------------------------------------------------------------------------------------------------------------------------


class VideoLog(object):
    def __init__(self, dataset_id, vlog_size, max_output_size=(1920, 1080)):
        self._vlog_size = vlog_size
        file_ac = FileAccess(storage="extract", resource_id=dataset_id)
        self._dest_path = file_ac.get_local_folder()
        self._dataset_id = dataset_id
        self._max_output_size = max_output_size
        self._last_log = datetime.datetime(2000, 1, 1)


    def log_batch(self, image_batch, output_batch, annotations):
        for idx, image in enumerate(image_batch):
            if random.random() < float(self._vlog_size):
                obj_id = str(ObjectId())

                filename = obj_id + '.jpg'
                file_thumb = filename[:-4] + "_thumb.jpg"
                img = image["input_image"]
                if max(img.shape) > max(self._max_output_size):
                    img, _ = img_utils.resize_image_scale(img, max(self._max_output_size))

                cv2.imwrite(os.path.join(self._dest_path, filename), img)
                file_stat_img = os.stat(os.path.join(self._dest_path, filename))

                if max(img.shape) > THUMB_SIZE:
                    img_thumb, _ = img_utils.resize_image_scale(img, THUMB_SIZE)
                    cv2.imwrite(os.path.join(self._dest_path, file_thumb), img_thumb)
                else:
                    cv2.imwrite(os.path.join(self._dest_path, file_thumb), img)
                file_stat_thumb = os.stat(os.path.join(self._dest_path, file_thumb))

                img_data = {
                    "_id": obj_id,
                    "date": pytz.utc.localize(datetime.datetime.now()),
                    "img_height": img.shape[0],
                    "img_width": img.shape[1],
                    "file_size": file_stat_img.st_size,
                    "thumb_size": file_stat_thumb.st_size,
                    "annotations": annotations[idx]
                }

                if 'frame_time' in image["frame_data"]:
                    img_data['frame_time'] = image["frame_data"]['frame_time']

                if 'video_data' in image["frame_data"]:
                    img_data['video_data'] = image["frame_data"]['video_data']

                with open(os.path.join(self._dest_path, obj_id + '_data.json'), 'w', newline='', encoding='utf8') as file_p:
                    json.dump(img_data, file_p, ensure_ascii=False, default=str)

                if (datetime.datetime.now() - self._last_log) > datetime.timedelta(minutes=1):
                    clear_log(self._dest_path)
                    self._last_log = datetime.datetime.now()
#----------------------------------------------------------------------------------------------------------------------------------
