"""
SiliconLife Eyeflow
WS-Edge client functions

Author: Alex Sobral de Freitas
"""

import os
from subprocess import call
from pathlib import Path
import datetime
import json
import cv2
import requests
import tarfile
import jwt

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


def get_model(app_token, dataset_id, model_folder):
    local_doc = None
    try:
        log.info(f"Get model {dataset_id}")

        folder_path = Path(model_folder + '/' + dataset_id)
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True, exist_ok=True)

        local_cache = os.path.join(model_folder, dataset_id + '.json')
        if os.path.isfile(local_cache):
            with open(local_cache) as fp:
                local_doc = json.load(fp)

        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        url = f"{endpoint}/published-model/{dataset_id}/"
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        payload = {"download_url": False}
        response = requests.get(url, headers=msg_headers, params=payload)

        if response.status_code != 200:
            if local_doc:
                return local_doc

            log.error(f"Failing get model: {response.json()}")
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

        dest_filename = os.path.join(model_folder, dataset_id + ".tar.gz")
        download_file(model_doc["download_url"], dest_filename)

        # expand_file
        call([
            "tar",
            "-xzf", dest_filename,
            "--directory", folder_path
        ])

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


def upload_model(
    app_token,
    dataset_id,
    model_info,
    model_folder,
    train_id,
    train_info,
    hist_folder
):
    try:
        model_filename = os.path.join(model_folder, dataset_id + ".tar.gz")
        if os.path.isfile(model_filename):
            os.remove(model_filename)

        folder_path = os.path.join(model_folder, dataset_id)
        wd = os.getcwd()
        os.chdir(folder_path)
        files_list = get_list_files_info("./")
        with tarfile.open(model_filename, "w:gz") as tar:
            for filename in files_list:
                tar.add(filename)

        os.chdir(wd)

        hist_filename = os.path.join(hist_folder, train_id + ".tar.gz")
        if os.path.isfile(hist_filename):
            os.remove(hist_filename)

        wd = os.getcwd()
        os.chdir(hist_folder)
        files_list = get_list_files_info("./")
        with tarfile.open(hist_filename, "w:gz") as tar:
            for filename in files_list:
                tar.add(filename)

        os.chdir(wd)

        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        url = f"{endpoint}/model/{dataset_id}/{train_id}"

        files = {
            'model_file': open(model_filename, 'rb'),
            'train_file': open(hist_filename, 'rb')
        }

        test_batch_filename = os.path.join(hist_folder, "test_batch-" + dataset_id + ".jpg")
        if os.path.isfile(test_batch_filename):
            files['test_batch'] = open(test_batch_filename, 'rb')

        test_augmentation_filename = os.path.join(hist_folder, "test_augmentation-" + dataset_id + ".jpg")
        if os.path.isfile(test_augmentation_filename):
            files['test_augmentation'] = open(test_augmentation_filename, 'rb')

        model_info["size"] = os.stat(model_filename).st_size
        train_info["size"] = os.stat(hist_filename).st_size
        values = {
            'model_info': json.dumps(model_info, default=str),
            'train_info': json.dumps(train_info, default=str)
        }

        response = requests.post(url, files=files, data=values, headers=msg_headers)

        if response.status_code != 201:
            raise Exception(f"Failing upload model: {response.json()}")

        os.remove(model_filename)
        os.remove(hist_filename)

        return dataset_id

    except requests.ConnectionError as error:
        log.error(f'Failing uploading model: {dataset_id}. Connection error: {error}')
        return None
    except requests.Timeout as error:
        log.error(f'Failing uploading model: {dataset_id}. Timeout: {error}')
        return None
    except Exception as excp:
        log.error(f'Failing uploading model: {dataset_id} - {excp}')
        return None
# ---------------------------------------------------------------------------------------------------------------------------------


def get_train(app_token, dataset_id, train_id, train_folder):
    try:
        log.info(f"Get train {dataset_id}-{train_id}")
        folder_path = Path(train_folder + '/' + dataset_id + '/' + train_id)
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True, exist_ok=True)

        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        url = f"{endpoint}/model-hist/{dataset_id}/{train_id}"
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        payload = {"download_url": True}
        response = requests.get(url, headers=msg_headers, params=payload)

        if response.status_code != 200:
            log.error(f"Failing get train: {response.json()}")
            return None

        train_doc = response.json()

        dest_filename = os.path.join(str(folder_path), train_id + ".tar.gz")
        download_file(train_doc["download_url"], dest_filename)

        # expand_file
        call([
            "tar",
            "-xf", dest_filename,
            "--directory", str(folder_path)
        ])

        os.remove(dest_filename)
        return train_id

    except requests.ConnectionError as error:
        log.error(f'Failing get train_id: {train_id}. Connection error: {error}')
        return None
    except requests.Timeout as error:
        log.error(f'Failing get train_id: {train_id}. Timeout: {error}')
        return None
    except Exception as excp:
        log.error(f'Failing get train_id: {train_id} - {excp}')
        return None
# ---------------------------------------------------------------------------------------------------------------------------------


def insert_train_event(app_token, event):
    try:
        endpoint = jwt.decode(app_token, options={"verify_signature": False})['endpoint']
        msg_headers = {'Authorization' : f'Bearer {app_token}'}
        url = f"{endpoint}/train/event"

        data = {
            "event": json.dumps(event, default=str)
        }
        response = requests.post(url, data=data, headers=msg_headers)

        if response.status_code != 201:
            raise Exception(f"Failing insert event: {response.json()}")

        return True

    except requests.ConnectionError as error:
        log.error(f'Failing inserting train event. Connection error: {error}')
        return None
    except requests.Timeout as error:
        log.error(f'Failing inserting train event. Timeout: {error}')
        return None
    except Exception as excp:
        log.error(f'Failing inserting train event - {excp}')
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
        call([
            "tar",
            "-xf", dest_filename,
            "--directory", folder_path
        ])

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
        call([
            "tar",
            "-xf", dest_filename,
            "--directory", folder_path
        ])

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


def generate_extract_thumbs(extract_path, thumb_size):
    """ Generate thumb image for all image files in extract folder
    """
    thumbs_list = [fname for fname in os.listdir(extract_path) if fname.endswith('_thumb.jpg')]
    for filename in os.listdir(extract_path):
        file_thumb = filename[:-4] + "_thumb.jpg"
        if filename.endswith('.jpg') and filename not in thumbs_list and file_thumb not in thumbs_list:
            img = cv2.imread(os.path.join(extract_path, filename))
            if max(img.shape) > thumb_size:
                img, _ = resize_image_scale(img, thumb_size)
            cv2.imwrite(os.path.join(extract_path, file_thumb), img)
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
        generate_extract_thumbs(folder_path, thumb_size)

        files_data = []
        file_list = [f for f in os.listdir(folder_path)]
        for filename in file_list:
            exp_id = filename[:24]
            if filename.endswith('_data.json') and (exp_id + ".jpg") in file_list and (exp_id + "_thumb.jpg") in file_list:
                try:
                    filepath = os.path.join(folder_path, filename)
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

        with open(os.path.join(folder_path, 'extract_files.json'), 'w', newline='', encoding='utf8') as file_p:
            json.dump(extract_files, file_p, ensure_ascii=False, default=str)

        dest_filename = os.path.join(extract_folder, dataset_id + ".tar.gz")
        if os.path.isfile(dest_filename):
            os.remove(dest_filename)

        wd = os.getcwd()
        os.chdir(folder_path)
        with tarfile.open(dest_filename, "w:gz") as tar:
            for filename in os.listdir(folder_path):
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
            raise Exception(f"Failing upload extract files: {response.json()}")

        os.remove(dest_filename)
        return dataset_id

    except requests.ConnectionError as error:
        log.error(f'Failing post upload_extract: {dataset_id}. Connection error: {error}')
        return None
    except requests.Timeout as error:
        log.error(f'Failing post upload_extract: {dataset_id}. Timeout: {error}')
        return None
    except Exception as excp:
        log.error(f'Failing post upload_extract: {dataset_id} - {excp}')
        return None
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
