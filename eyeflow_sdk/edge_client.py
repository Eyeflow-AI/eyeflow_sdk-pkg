"""
SiliconLife Eyeflow
WS-Edge client functions

Author: Alex Sobral de Freitas
"""

import os
from pathlib import Path
import datetime
import json
import cv2
import requests
import shutil
import tarfile
import jwt
import traceback

from eyeflow_sdk.log_obj import CONFIG, log
from eyeflow_sdk.img_utils import resize_image_scale
# ---------------------------------------------------------------------------------------------------------------------------------


def get_list_files_info(folder):
    file_list = []
    for filename in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, filename)):
            file_list.append(filename)
        elif os.path.isdir(os.path.join(folder, filename)):
            subfolder = os.path.join(folder, filename)
            subfolder_files = get_list_files_info(subfolder)
            for filename in subfolder_files:
                filename = os.path.join(os.path.split(subfolder)[1], filename)
                file_list.append(filename)

    return file_list
#----------------------------------------------------------------------------------------------------------------------------------


def download_file(url, local_filename):
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        if os.path.isfile(local_filename):
            os.remove(local_filename)

        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment 'if' and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
# ---------------------------------------------------------------------------------------------------------------------------------


def get_dataset(app_token, dataset_id):
    try:
        log.info(f"Get dataset {dataset_id}")

        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        response = requests.get(f"{endpoint}/dataset/{dataset_id}", headers=msg_headers)

        if response.status_code != 200:
            log.error(f"Failing get dataset: {response.json()}")
            return None

        dataset = response.json()
        if dataset["dataset_parms"]:
            return dataset
        else:
            log.warning(f"Failing get dataset: {response.json()}")
            return None

    except requests.ConnectionError as error:
        log.error(f'Failing get dataset_id: {dataset_id}. Connection error: {error}')
        return None
    except requests.Timeout as error:
        log.error(f'Failing get dataset_id: {dataset_id}. Timeout: {error}')
        return None
    except Exception as excp:
        log.error(f'Failing get dataset_id: {dataset_id} - {excp}')
        return None
# ---------------------------------------------------------------------------------------------------------------------------------


def get_flow(app_token, flow_id):
    try:
        log.info(f"Get flow {flow_id}")

        if not Path(CONFIG["flow_folder"]).is_dir():
            Path(CONFIG["flow_folder"]).mkdir(parents=True, exist_ok=True)

        local_cache = os.path.join(CONFIG["flow_folder"], flow_id + '.json')

        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        response = requests.get(f"{endpoint}/flow/{flow_id}", headers=msg_headers)

        if response.status_code != 200:
            log.error(f"Failing get flow from edge: {response.json()}")

            if os.path.isfile(local_cache):
                with open(local_cache) as fp:
                    flow = json.load(fp)

                return flow

            return None

        flow = response.json()["flow"]
        if "_id" in flow:

            if os.path.isfile(local_cache):
                os.remove(local_cache)

            with open(local_cache, 'w') as fp:
                json.dump(flow, fp, default=str)

            return flow
        else:
            log.warning(f"Failing get flow: {response.json()}")
            return None

    except requests.ConnectionError as error:
        log.error(f'Failing get flow_id: {flow_id}. Connection error: {error}')
        return None
    except requests.Timeout as error:
        log.error(f'Failing get flow_id: {flow_id}. Timeout: {error}')
        return None
    except Exception as excp:
        log.error(f'Failing get flow_id: {flow_id} - {excp}')
        return None
# ---------------------------------------------------------------------------------------------------------------------------------


def get_model(app_token, dataset_id, model_folder, model_type="tensorflow"):
    local_doc = None
    try:
        # log.info(f"Check model {dataset_id}")

        local_cache = os.path.join(model_folder, dataset_id + '.json')
        if os.path.isfile(local_cache):
            with open(local_cache) as fp:
                local_doc = json.load(fp)

        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        url = f"{endpoint}/published-model-v2/{dataset_id}/"
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        payload = {"download_url": False}
        response = requests.get(url, headers=msg_headers, params=payload)

        if response.status_code != 200:
            if local_doc:
                return local_doc

            log.error(f"Failing get model: {url} - {response.json()}")
            return None

        model_doc = response.json()
        if local_doc and model_doc["date"] == local_doc["date"]:
            return local_doc

        payload = {"download_url": True}
        response = requests.get(url, headers=msg_headers, params=payload)

        if response.status_code != 200:
            if local_doc:
                return local_doc

            log.error(f"Failing get model: {response.json()}")
            return None

        model_doc = response.json()

        if "model_list" not in model_doc:
            log.error(f"Get model response dont have model_list key: {model_doc}")
            raise Exception(f"Get model response dont have model_list key: {model_doc}")

        for model_data in model_doc["model_list"]:
            if model_data.get("type", "") == model_type:
                download_url = model_data["download_url"]
                dest_filename = os.path.join(model_folder, model_data["file"])
                break
        else:
            log.error(f"Did not find model type {model_type} in {dataset_id} document - {model_doc}")
            return model_doc

        log.info(f"Download model {dataset_id} - Train date: {model_doc['date']}")
        download_file(download_url, dest_filename)

        # expand_file
        if (dest_filename.endswith('tar.gz')):

            folder_path = Path(model_folder + '/' + dataset_id)
            if not folder_path.is_dir():
                folder_path.mkdir(parents=True, exist_ok=True)

            with tarfile.open(dest_filename, 'r') as tar:
                tar.extractall(folder_path)

            os.remove(dest_filename)

        if os.path.isfile(local_cache):
            os.remove(local_cache)

        with open(local_cache, 'w') as fp:
            json.dump(model_doc, fp, default=str)

        return model_doc

    except requests.ConnectionError as error:
        if local_doc:
            return local_doc

        log.error(f'Failing get model dataset_id: {dataset_id}. Connection error: {error}')
        return None
    except requests.Timeout as error:
        if local_doc:
            return local_doc

        log.error(f'Failing get model dataset_id: {dataset_id}. Timeout: {error}')
        return None
    except Exception as excp:
        log.error(f'Failing get model dataset_id: {dataset_id} - {excp}')
        return None
# ---------------------------------------------------------------------------------------------------------------------------------


def get_flow_component(app_token, flow_component_id, flow_component_folder):
    local_doc = None
    try:
        log.info(f"Get flow_component {flow_component_id}")

        folder_path = Path(flow_component_folder + '/' + flow_component_id)
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True, exist_ok=True)

        local_cache = os.path.join(flow_component_folder, flow_component_id + '.json')
        if os.path.isfile(local_cache):
            with open(local_cache) as fp:
                local_doc = json.load(fp)

        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        url = f"{endpoint}/flow-component/{flow_component_id}"
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        payload = {"download_url": False}
        response = requests.get(url, headers=msg_headers, params=payload)

        if response.status_code != 200:
            if local_doc:
                return local_doc

            log.error(f"Failing get flow-component: {response.json()}")
            return None

        flow_component_doc = response.json()
        if local_doc and flow_component_doc["version"] == local_doc["version"]:
            return local_doc

        payload = {"download_url": True}
        response = requests.get(url, headers=msg_headers, params=payload)

        if response.status_code != 200:
            if local_doc:
                return local_doc

            log.error(f"Failing get model: {response.json()}")
            return None

        flow_component_doc = response.json()

        dest_filename = os.path.join(folder_path, flow_component_id + ".tar.gz")
        download_file(flow_component_doc["download_url"], dest_filename)

        # expand_file
        with tarfile.open(dest_filename, 'r') as tar:
            tar.extractall(folder_path)

        os.remove(dest_filename)

        if os.path.isfile(local_cache):
            os.remove(local_cache)

        with open(local_cache, 'w') as fp:
            json.dump(flow_component_doc, fp, default=str)

        return flow_component_doc

    except requests.ConnectionError as error:
        if local_doc:
            return local_doc

        log.error(f'Failing get flow_component: {flow_component_id}. Connection error: {error}')
        return None
    except requests.Timeout as error:
        if local_doc:
            return local_doc

        log.error(f'Failing get flow_component: {flow_component_id}. Timeout: {error}')
        return None
    except Exception as excp:
        if local_doc:
            return local_doc

        log.error(f'Failing get flow_component: {flow_component_id} - {excp}')
        return None
# ---------------------------------------------------------------------------------------------------------------------------------


def get_model_component(app_token, model_component_id, model_component_folder):
    local_doc = None
    try:
        log.info(f"Get model_component {model_component_id}")
        folder_path = Path(model_component_folder + '/' + model_component_id)
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True, exist_ok=True)

        local_cache = os.path.join(model_component_folder, model_component_id + '.json')
        if os.path.isfile(local_cache):
            with open(local_cache) as fp:
                local_doc = json.load(fp)

        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        url = f"{endpoint}/model-component/{model_component_id}"
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        payload = {"download_url": False}
        response = requests.get(url, headers=msg_headers, params=payload)

        if response.status_code != 200:
            if local_doc:
                return local_doc

            log.error(f"Failing get model_component: {response.json()}")
            return None

        model_component_doc = response.json()
        if local_doc and model_component_doc["version"] == local_doc["version"]:
            return local_doc

        payload = {"download_url": True}
        response = requests.get(url, headers=msg_headers, params=payload)

        if response.status_code != 200:
            if local_doc:
                return local_doc

            log.error(f"Failing get model: {response.json()}")
            return None

        model_component_doc = response.json()

        dest_filename = os.path.join(folder_path, model_component_id + ".tar.gz")
        download_file(model_component_doc["download_url"], dest_filename)

        # expand_file
        with tarfile.open(dest_filename, 'r') as tar:
            tar.extractall(folder_path)

        os.remove(dest_filename)

        if os.path.isfile(local_cache):
            os.remove(local_cache)

        with open(local_cache, 'w') as fp:
            json.dump(model_component_doc, fp, default=str)

        return model_component_id

    except requests.ConnectionError as error:
        if local_doc:
            return local_doc

        log.error(f'Failing get model_component: {model_component_id}. Connection error: {error}')
        return None
    except requests.Timeout as error:
        if local_doc:
            return local_doc

        log.error(f'Failing get model_component: {model_component_id}. Timeout: {error}')
        return None
    except Exception as excp:
        if local_doc:
            return local_doc

        log.error(f'Failing get model_component: {model_component_id} - {excp}')
        return None
# ---------------------------------------------------------------------------------------------------------------------------------


def clear_log(extract_path, max_files):
    files_list = [fname for fname in os.listdir(extract_path) if fname.endswith('.jpg') and not fname.endswith('_thumb.jpg')]
    if len(files_list) > max_files:
        date_list = [(filename, datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(extract_path, filename)))) for filename in files_list]
        exclude_list = sorted(date_list, key=lambda x: x[1])[:len(files_list) - max_files]
        for filename, _ in exclude_list:
            try:
                os.remove(os.path.join(extract_path, filename))
                os.remove(os.path.join(extract_path, filename[:-4] + '_data.json'))
                os.remove(os.path.join(extract_path, filename[:-4] + '_thumb.jpg'))
            except:
                pass
#----------------------------------------------------------------------------------------------------------------------------------


def generate_images_thumb(images_path, thumb_size):
    """ Generate thumb image for all image files in {images_path}
    """
    thumbs_list = [fname for fname in os.listdir(images_path) if fname.endswith('_thumb.jpg')]
    for filename in os.listdir(images_path):
        file_thumb = filename[:-4] + "_thumb.jpg"
        if filename.endswith('.jpg') and filename not in thumbs_list and file_thumb not in thumbs_list:
            try:
                img = cv2.imread(os.path.join(images_path, filename))
                if max(img.shape) > thumb_size:
                    img, _ = resize_image_scale(img, thumb_size)
                cv2.imwrite(os.path.join(images_path, file_thumb), img)
            except:
                log.error(f"Fail to generate image thumb: {filename}. Removing")
                os.remove(os.path.join(images_path, filename))
#----------------------------------------------------------------------------------------------------------------------------------


MAX_EXTRACT_FILES = 400
THUMB_SIZE = 128
def upload_extract(app_token, dataset_id, extract_folder, max_files=MAX_EXTRACT_FILES, thumb_size=THUMB_SIZE):
    try:
        log.info(f"Upload extract {dataset_id}")
        folder_path = os.path.join(extract_folder, dataset_id)
        if not os.path.isdir(folder_path):
            raise Exception(f"Extract folder doesn't exists: {folder_path}")

        clear_log(folder_path, max_files)

        tmp_path = os.path.join(extract_folder, "tmp", dataset_id)
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path)

        shutil.copytree(folder_path, tmp_path)

        generate_images_thumb(tmp_path, thumb_size)

        files_data = []
        file_list = [f for f in os.listdir(tmp_path)]
        for filename in file_list:
            exp_id = filename[:24]
            if filename.endswith('_data.json') and (exp_id + ".jpg") in file_list and (exp_id + "_thumb.jpg") in file_list:
                try:
                    filepath = os.path.join(tmp_path, filename)
                    with open(filepath, 'r') as json_file:
                        data = json.load(json_file)

                        if "date" in data:
                            data["date"] = {"$date": data["date"]}

                        if "_id" in data:
                            data["_id"] = {"$oid": data["_id"]}

                        files_data.append(data)
                except:
                    pass

        if not files_data:
            log.warning(f'Cannot upload post upload_extract: {dataset_id}. No files.')
            return dataset_id

        extract_files = {
            "files_data": files_data
        }

        with open(os.path.join(tmp_path, 'extract_files.json'), 'w', newline='', encoding='utf8') as file_p:
            json.dump(extract_files, file_p, ensure_ascii=False, default=str)

        dest_filename = os.path.join(extract_folder, dataset_id + ".tar.gz")
        if os.path.isfile(dest_filename):
            os.remove(dest_filename)

        wd = os.getcwd()
        os.chdir(tmp_path)
        with tarfile.open(dest_filename, "w:gz") as tar:
            for filename in os.listdir(tmp_path):
                if filename.endswith('.jpg') or filename == 'extract_files.json':
                    tar.add(filename)

        os.chdir(wd)

        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        url = f"{endpoint}/dataset/{dataset_id}/extract"

        files = {'extract': open(dest_filename, 'rb')}
        values = {
            'dataset_id': dataset_id,
            'extract_files': extract_files
        }

        response = requests.post(url, files=files, data=values, headers=msg_headers)

        if response.status_code != 201:
            raise Exception(f"Failing upload extract files: {response}")

        os.remove(dest_filename)
        shutil.rmtree(tmp_path)

        return dataset_id

    except requests.ConnectionError as error:
        log.error(f'Failing post upload_extract: {dataset_id}. Connection error: {error}')
        return None
    except requests.Timeout as error:
        log.error(f'Failing post upload_extract: {dataset_id}. Timeout: {error}')
        return None
    except Exception as excp:
        log.error(f'Failing post upload_extract: {dataset_id} - {excp}')
        log.error(traceback.format_exc())
        return None
#----------------------------------------------------------------------------------------------------------------------------------

def upload_feedback(app_token, dataset_id, feedback_folder, thumb_size=THUMB_SIZE):
    try:
        log.info(f"Upload feedback {dataset_id}")
        folder_path = os.path.join(feedback_folder, dataset_id)
        if not os.path.isdir(folder_path):
            raise Exception(f"Feedback folder doesn't exists: {folder_path}")

        generate_images_thumb(folder_path, thumb_size)

        file_list = list(os.listdir(folder_path))
        json_file_list = [i for i in file_list if i.endswith('_data.json')]
        tar_files_list = []
        feedback_files = []
        for json_filename in json_file_list:
            exp_id = json_filename[:24]
            image_filename = f"{exp_id}.jpg"
            image_thumb_filename = f"{exp_id}_thumb.jpg"
            if image_filename in file_list and image_thumb_filename in file_list:
                try:
                    filepath = os.path.join(folder_path, json_filename)
                    with open(filepath, 'r') as json_file:
                        data = json.load(json_file)

                        if "_id" in data:
                            data["_id"] = {"$oid": data["_id"]}

                        if "date" in data:
                            data["date"] = {"$date": data["date"]}

                        if "dataset_id" in data:
                            data["dataset_id"] = {"$oid": data["dataset_id"]}

                        feedback_files.append(data)
                        tar_files_list += [
                            image_filename,
                            image_thumb_filename
                        ]
                except:
                    pass

        if not tar_files_list:
            log.warning(f'Cannot upload post upload_feedback: {dataset_id}. No files.')
            return dataset_id

        dest_filename = os.path.join(feedback_folder, dataset_id + ".tar.gz")
        if os.path.isfile(dest_filename):
            os.remove(dest_filename)

        wd = os.getcwd()
        os.chdir(folder_path)
        with tarfile.open(dest_filename, "w:gz") as tar:
            for filename in tar_files_list:
                if filename.endswith('.jpg') or filename:
                    tar.add(filename)

        os.chdir(wd)

        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        url = f"{endpoint}/dataset/{dataset_id}/feedback"

        files = {
            'feedback_files': json.dumps(feedback_files),
            'feedback': open(dest_filename, 'rb')
            }

        response = requests.post(url, files=files, headers=msg_headers)

        if response.status_code != 201:
            raise Exception(f"Failing upload extract files: {response.json()}")

        os.remove(dest_filename)
        return True

    except requests.ConnectionError as error:
        log.error(f'Failing post upload_feedback: {dataset_id}. Connection error: {error}')
        return False
    except requests.Timeout as error:
        log.error(f'Failing post upload_feedback: {dataset_id}. Timeout: {error}')
        return False
    except Exception as excp:
        log.error(f'Failing post upload_feedback: {dataset_id} - {excp}')
        return False
# ---------------------------------------------------------------------------------------------------------------------------------


def get_video(app_token, video_id, video_folder):
    local_doc = None
    try:
        folder_path = Path(video_folder)
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True, exist_ok=True)

        local_cache = os.path.join(video_folder, video_id + '.json')
        if os.path.isfile(local_cache):
            with open(local_cache) as fp:
                local_doc = json.load(fp)

        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        url = f"{endpoint}/video/{video_id}"
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        payload = {"download_url": False}
        response = requests.get(url, headers=msg_headers, params=payload)

        if response.status_code != 200:
            if local_doc:
                return local_doc

            log.error(f"Failing get video: {response.json()}")
            return None

        video_doc = response.json()
        if local_doc and video_doc["date"] == local_doc["date"]:
            return local_doc

        payload = {"download_url": True}
        response = requests.get(url, headers=msg_headers, params=payload)

        if response.status_code != 200:
            if local_doc:
                return local_doc

            log.error(f"Failing get model: {response.json()}")
            return None

        video_doc = response.json()

        dest_filename = os.path.join(video_folder, video_id + ".mp4")
        download_file(video_doc["download_url"], dest_filename)

        if os.path.isfile(local_cache):
            os.remove(local_cache)

        with open(local_cache, 'w') as fp:
            json.dump(video_doc, fp, default=str)

        return video_doc

    except requests.ConnectionError as error:
        if local_doc:
            return local_doc

        log.error(f'Failing get video: {video_id}. Connection error: {error}')
        return None
    except requests.Timeout as error:
        if local_doc:
            return local_doc

        log.error(f'Failing get video: {video_id}. Timeout: {error}')
        return None
    except Exception as excp:
        if local_doc:
            return local_doc

        log.error(f'Failing get video: {video_id} - {excp}')
        return None
# ---------------------------------------------------------------------------------------------------------------------------------


def upload_video(app_token, video_id, video_file_annotation, video_file, output_file, video_folder):
    try:
        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        url = f"{endpoint}/video/{video_id}"

        video_filename = os.path.join(video_folder, video_file)
        output_filename = os.path.join(video_folder, output_file)
        files = {
            'video': open(video_filename, 'rb'),
            'output': open(output_filename, 'rb')
        }
        values = {
            "annotations": json.dumps(video_file_annotation, default=str)
        }

        response = requests.post(url, files=files, data=values, headers=msg_headers)

        if response.status_code != 201:
            raise Exception(f"Failing upload video: {response.json()}")

        return video_id

    except requests.ConnectionError as error:
        log.error(f'Failing uploading video: {video_file}-{video_id}. Connection error: {error}')
        return None
    except requests.Timeout as error:
        log.error(f'Failing uploading video: {video_file}-{video_id}. Timeout: {error}')
        return None
    except Exception as excp:
        log.error(f'Failing uploading video: {video_file}-{video_id} - {excp}')
        return None
# ---------------------------------------------------------------------------------------------------------------------------------


def get_edge_data(app_token):
    try:
        log.info(f"Get edge_data")
        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        response = requests.get(f"{endpoint}", headers=msg_headers)

        if response.status_code != 200:
            log.error(f"Failing get edge_data: {response.json()}")
            return None

        return response.json()["edge_data"]

    except requests.ConnectionError as error:
        log.error(f'Failing get edge_data. Connection error: {error}')
        return None
    except requests.Timeout as error:
        log.error(f'Failing get edge_data. Timeout: {error}')
        return None
    except Exception as excp:
        log.error(f'Failing get edge_data - {excp}')
        return None
#----------------------------------------------------------------------------------------------------------------------------------
