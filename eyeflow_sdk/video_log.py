"""
SiliconLife Eyeflow
Class for log batch of extracted images from detection

Author: Alex Sobral de Freitas
"""

import os

import json
import datetime
import random
from bson import ObjectId
import cv2

from eyeflow_sdk.file_access import FileAccess
from eyeflow_sdk.log_obj import log
#----------------------------------------------------------------------------------------------------------------------------------

MAX_EXTRACT_FILES = 800

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

def upload_extracts(dataset_name, cloud_parms):
    """
    Upload extracts of process to cloud
    """

    log.info(f"Upload extracts dataset: {dataset_name}")

    def save_extract_list(extract_path):

        date_list = []
        for filename in os.listdir(extract_path):
            if filename.endswith('_data.json'):
                try:
                    filepath = os.path.join(extract_path, filename)
                    with open(filepath, 'r') as json_file:
                        data = json.load(json_file)
                        if 'frame_time' in data:
                            frame_time = data['frame_time']
                        else:
                            frame_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                        date_list.append([filename, frame_time])
                except:
                    pass

        files_list = sorted(date_list, key=lambda x: x[1], reverse=True)
        with open(os.path.join(extract_path, 'extract_files.json'), 'w', newline='', encoding='utf8') as file_p:
            json.dump({"extract_list": files_list}, file_p, ensure_ascii=False, indent=2, default=str)

    file_ac = FileAccess(storage="extract", resource_id=dataset_name, cloud_parms=cloud_parms)
    clear_log(file_ac.get_local_folder())
    save_extract_list(file_ac.get_local_folder())
    file_ac.sync_files(origin="local")
#----------------------------------------------------------------------------------------------------------------------------------

class VideoLog(object):
    def __init__(self, dataset_name, vlog_size):
        self._vlog_size = vlog_size
        file_ac = FileAccess(storage="extract", resource_id=dataset_name)
        self._dest_path = file_ac.get_local_folder()
        self._last_log = datetime.datetime(2000, 1, 1)


    def log_batch(self, image_batch, output_batch, annotations):
        for idx, image in enumerate(image_batch):
            if random.random() < float(self._vlog_size):
                obj_id = str(ObjectId())
                cv2.imwrite(os.path.join(self._dest_path, obj_id + '.jpg'), image[0])
                img_data = {
                    "_id": obj_id,
                    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "img_height": image[0].shape[0],
                    "img_width": image[0].shape[1],
                    "detections": annotations[idx],
                    "annotations": annotations[idx]
                }

                if 'frame_time' in image[2]:
                    img_data['frame_time'] = image[2]['frame_time']

                with open(os.path.join(self._dest_path, obj_id + '_data.json'), 'w', newline='', encoding='utf8') as file_p:
                    json.dump(img_data, file_p, ensure_ascii=False, indent=2, default=str)

                if (datetime.datetime.now() - self._last_log) > datetime.timedelta(minutes=1):
                    clear_log(self._dest_path)
                    self._last_log = datetime.datetime.now()
#----------------------------------------------------------------------------------------------------------------------------------
