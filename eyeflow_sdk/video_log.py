"""
SiliconLife Eyeflow
Class for log batch of extracted images from detection

Author: Alex Sobral de Freitas
"""

import os

import json
import datetime
import random
import cv2
import importlib

from pymongo import MongoClient
from bson import ObjectId

from eyeflow_sdk.file_access import FileAccess
from eyeflow_sdk.img_utils import resize_image_scale
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

def upload_extracts(dataset_id, db_config, cloud_parms):
    """
    Upload extracts of process to cloud
    """

    log.info(f"Upload extracts dataset: {dataset_id}")
    comp_lib = importlib.import_module(f'eyeflow_sdk.cloud_store.{cloud_parms["provider"]}')
    cloud_obj = comp_lib.Connector(**cloud_parms)


    def generate_extract_thumbs(extract_path):
        """ Generate thumb image for all image files in extract folder
        """
        thumbs_list = [fname for fname in os.listdir(extract_path) if fname.endswith('_thumb.jpg')]
        for filename in os.listdir(extract_path):
            if filename.endswith('.jpg') and filename not in thumbs_list:
                file_thumb = filename[:-4] + "_thumb.jpg"
                img = cv2.imread(os.path.join(extract_path, filename))
                if max(img.shape) > THUMB_SIZE:
                    img, _ = resize_image_scale(img, THUMB_SIZE)
                cv2.imwrite(os.path.join(extract_path, file_thumb), img)


    def save_extract_list(extract_path):
        """ Save a json with info about all files in extract folder
        """
        extract_files = {
            "files_data": []
        }

        files_list = []
        for filename in os.listdir(extract_path):
            if filename.endswith('_data.json'):
                try:
                    filepath = os.path.join(extract_path, filename)
                    with open(filepath, 'r') as json_file:
                        data = json.load(json_file)
                        extract_files["files_data"].append(data)
                        if 'date' in data:
                            file_time = data['date']
                        else:
                            file_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M:%S.%f")
                        files_list.append([filename, file_time])
                except:
                    pass

        cloud_files = cloud_obj.list_files_info(folder="extract", resource_id=dataset_id)
        for cloud_file in cloud_files:
            if cloud_file["filename"].endswith('_data.json'):
                try:
                    data = json.loads(cloud_obj.download_file(folder="extract", resource_id=dataset_id, filename=cloud_file["filename"]))
                    extract_files["files_data"].append(data)
                    if 'date' in data:
                        file_time = data['date']
                    else:
                        file_time = cloud_file["creation_date"].strftime("%Y-%m-%d %H:%M:%S.%f")
                    files_list.append([filename, file_time])
                except:
                    pass

        extract_files["extract_list"] = sorted(files_list, key=lambda x: x[1], reverse=True)

        # save extract info in storage
        with open(os.path.join(extract_path, 'extract_files.json'), 'w', newline='', encoding='utf8') as file_p:
            json.dump(extract_files, file_p, ensure_ascii=False, default=str)

        # save extract info in database
        client = MongoClient(db_config["db_url"])
        db_mongo = client[db_config["db_name"]]

        db_mongo.extract.delete_one({"_id": ObjectId(dataset_id)})

        extract_files["_id"] = ObjectId(dataset_id)
        db_mongo.extract.insert_one(extract_files)


    file_ac = FileAccess(storage="extract", resource_id=dataset_id, cloud_parms=cloud_parms)
    # clear_log(file_ac.get_local_folder())
    file_ac.purge_files(max_files=MAX_EXTRACT_FILES)
    generate_extract_thumbs(file_ac.get_local_folder())
    save_extract_list(file_ac.get_local_folder())
    file_ac.sync_files(origin="local")
#----------------------------------------------------------------------------------------------------------------------------------

class VideoLog(object):
    def __init__(self, dataset_id, vlog_size):
        self._vlog_size = vlog_size
        file_ac = FileAccess(storage="extract", resource_id=dataset_id)
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

                if 'video_file' in image[2]:
                    img_data['video_file'] = image[2]['video_file']

                with open(os.path.join(self._dest_path, obj_id + '_data.json'), 'w', newline='', encoding='utf8') as file_p:
                    json.dump(img_data, file_p, ensure_ascii=False, indent=2, default=str)

                if (datetime.datetime.now() - self._last_log) > datetime.timedelta(minutes=1):
                    clear_log(self._dest_path)
                    self._last_log = datetime.datetime.now()
#----------------------------------------------------------------------------------------------------------------------------------
