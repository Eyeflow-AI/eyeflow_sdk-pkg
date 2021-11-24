"""
SiliconLife Eyeflow
Dataset manipulation functions

Author: Alex Sobral de Freitas
"""

import os
import json
import random
import copy
import datetime
import pytz
import dateutil.parser
import requests

import cv2
import numpy as np
import h5py

from bson.objectid import ObjectId

from eyeflow_sdk.log_obj import log, CONFIG
import eyeflow_sdk.edge_client as edge_client
#----------------------------------------------------------------------------------------------------------------------------------

def default_json_converter(obj):
    if isinstance(obj, datetime.datetime):
        return obj.replace(tzinfo=datetime.timezone.utc).isoformat()
    else:
        return str(obj)
#----------------------------------------------------------------------------------------------------------------------------------


class Dataset():
    """
    Class to serialize and deserialize datasets to/from a file
    """
    def __init__(self, dataset_id, app_token=None):
        self.id = str(ObjectId(dataset_id))
        self.dataset_name = None
        self._app_token = app_token

        self.parms = {}
        self.examples = []
        self.images = {}


    @staticmethod
    def get_dataset_default_parms():
        """
        Returns default dataset parms
        """
        filename = os.path.join(os.path.dirname(__file__), 'dataset_default_parms.json')
        with open(filename, 'r', newline='', encoding='utf8') as fp:
            default_parms = json.load(fp)

        return default_parms


    @staticmethod
    def get_data_augmentation_default_parms():
        """
        Returns default data augmentation parms
        """
        filename = os.path.join(os.path.dirname(__file__), 'data_augmentation_default_parms.json')
        if not os.path.isfile(filename):
            return None

        with open(filename, 'r', newline='', encoding='utf8') as fp:
            default_parms = json.load(fp)

        return default_parms


    @staticmethod
    def get_dataset_types():
        """
        Returns dataset types
        """
        filename = os.path.join(os.path.dirname(__file__), 'dataset_types.json')
        if not os.path.isfile(filename):
            return None

        with open(filename, 'r', newline='', encoding='utf8') as fp:
            default_parms = json.load(fp)

        return default_parms


    def update_default_parms(self):
        """
        Performs an update in parms with all default parms read from database
        """
        train_parms = {}
        dnn_parms = {}
        dataset_default_parms = Dataset.get_dataset_default_parms()
        if dataset_default_parms:
            train_parms = dataset_default_parms["default_parms"][self.parms["info"]["type"]]["train_parms"]
            dnn_parms = dataset_default_parms["default_parms"][self.parms["info"]["type"]]["dnn_parms"]

        input_shape = None
        if "dnn_parms" in self.parms:
            dnn_parms.update(self.parms["dnn_parms"])
            if "input_shape" in self.parms["dnn_parms"]:
                input_shape = copy.deepcopy(self.parms["dnn_parms"]["input_shape"])

            if "component" in self.parms["dnn_parms"] and "component_id" not in self.parms["dnn_parms"]:
                if self.parms["dnn_parms"]["component"] == "objdet_af":
                    dnn_parms["component_id"] = "6143a1faef5cc63fd4c177b1"
                elif self.parms["dnn_parms"]["component"] == "objdet":
                    dnn_parms["component_id"] = "6143a1edef5cc63fd4c177b0"
                elif self.parms["dnn_parms"]["component"] == "class_cnn":
                    dnn_parms["component_id"] = "614388073a692cccdab0e69b"
                elif self.parms["dnn_parms"]["component"] == "obj_location":
                    dnn_parms["component_id"] = "6178516681cbe716153175b0"

        elif "network_parms" in self.parms and "dnn_parms" in self.parms["network_parms"]:
            dnn_parms.update(self.parms["network_parms"]["dnn_parms"])

        if "train_parms" in self.parms:
            train_parms.update(self.parms["train_parms"])
            if not input_shape:
                input_shape = copy.deepcopy(self.parms["train_parms"]["input_shape"])
            if "input_shape" in train_parms:
                del train_parms["input_shape"]
        elif "network_parms" in self.parms:
            train_parms.update(self.parms["network_parms"])
            if "dnn_parms" in train_parms:
                del train_parms["dnn_parms"]

            if "input_shape" in train_parms:
                del train_parms["input_shape"]

        self.parms["train_parms"] = train_parms
        self.parms["dnn_parms"] = dnn_parms
        self.parms["dnn_parms"]["input_shape"] = input_shape

        if "network_parms" in self.parms:
            del self.parms["network_parms"]

        data_augmentation_parms = {}
        data_augmentation_default_parms = Dataset.get_data_augmentation_default_parms()
        if data_augmentation_default_parms:
            data_augmentation_parms = data_augmentation_default_parms["default_parms"]
            data_augmentation_parms.update(self.parms.get("data_augmentation_parms", {}))

        self.parms["data_augmentation_parms"] = data_augmentation_parms


    def load_data(self):
        """
        Load dataset data from ws
        """
        if not self._app_token:
            raise Exception('AppToken not set')

        dataset = edge_client.get_dataset(self._app_token, self.id)
        if not dataset:
            raise Exception(f'Fail loading dataset_id {self.id}')

        self.dataset_name = dataset["dataset_parms"].get("name")
        self.parms = dataset["dataset_parms"]
        self.examples = dataset["annotations"]
        self.update_default_parms()
        log.info(f"Load dataset from database {len(self.examples)} examples")


    @staticmethod
    def load_file_from_cloud(url):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            img = b""
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment 'if' and set chunk_size parameter to None.
                #if chunk:
                img += chunk

            return img


    def load_image(self, example_img):
        """
        Load image to memory from disk or cloud
        """
        for exp in self.examples:
            if exp["example"] == example_img:
                break
        else:
            raise Exception(f"Example not found {example_img}")

        exp_folder = os.path.join(CONFIG["file-service"]["dataset"]["local_folder"], self.id)
        if os.path.isfile(os.path.join(exp_folder, exp["example"])):
            with open(os.path.join(exp_folder, exp["example"]), 'rb') as fp:
                self.images[exp["example"]] = fp.read()
        else:
            self.images[exp["example"]] = self.load_file_from_cloud(exp["download_url"])


    def load_all_images(self):
        """
        Load images
        """
        for exp in self.examples:
            if exp["example"] not in self.images:
                exp_folder = os.path.join(CONFIG["file-service"]["dataset"]["local_folder"], self.id)
                if os.path.isfile(os.path.join(exp_folder, exp["example"])):
                    with open(os.path.join(exp_folder, exp["example"]), 'rb') as fp:
                        self.images[exp["example"]] = fp.read()
                else:
                    self.images[exp["example"]] = self.load_file_from_cloud(exp["download_url"])


    def save_images_to_disk(self):
        """
        Save images to disk
        """
        self.load_all_images()
        exp_folder = os.path.join(CONFIG["file-service"]["dataset"]["local_folder"], self.id)

        for exp in self.examples:
            if os.path.isfile(os.path.join(exp_folder, exp["example"])):
                os.remove(os.path.join(exp_folder, exp["example"]))

            with open(os.path.join(exp_folder, exp["example"]), 'wb') as fp:
                fp.write(self.images[exp["example"]])


    def get_example_img(self, example_img):
        """
        Returns an image in opencv format
        """
        if example_img not in self.images:
            self.load_image(example_img)

        return cv2.imdecode(np.frombuffer(self.images[example_img], dtype=np.uint8), cv2.IMREAD_COLOR)


    def get_train_subsets(self, shuffle=True):
        """
        Partitioning of dataset in 3 groups: train, validation and test
        """
        if not self.parms:
            raise Exception('Must load or import dataset first')

        train_dataset = Dataset(self.id, app_token=self._app_token)
        train_dataset.parms = copy.deepcopy(self.parms)

        val_dataset = Dataset(self.id, app_token=self._app_token)
        val_dataset.parms = copy.deepcopy(self.parms)

        test_dataset = Dataset(self.id, app_token=self._app_token)
        test_dataset.parms = copy.deepcopy(self.parms)

        if shuffle:
            random.shuffle(self.examples)

        val_size = int(len(self.examples) * self.parms["train_parms"]["val_size"])
        if val_size < 1:
            val_size = 1

        test_size = int(len(self.examples) * self.parms["train_parms"]["test_size"])
        if test_size < 1:
            test_size = 1

        if (val_size + test_size) > len(self.examples):
            raise Exception("Insufficient examples to train: %d" % len(self.examples))

        train_dataset.examples = copy.deepcopy(self.examples[test_size + val_size:])
        val_dataset.examples = copy.deepcopy(self.examples[test_size:test_size + val_size])
        test_dataset.examples = copy.deepcopy(self.examples[:test_size])

        return train_dataset, val_dataset, test_dataset


    @staticmethod
    def export_to_hdf5(export_data, parms, examples, images, filename, dataset_id):
        """
        Export dataset to file in hdf5 format
        """
        example_list = []
        image_list = []
        for exp in examples:
            example_list.append(json.dumps(exp, ensure_ascii=False, default=default_json_converter))
            image_list.append(images[exp["example"]])

        dt = h5py.special_dtype(vlen=str)
        example_list = np.array(example_list, dtype=dt)
        image_list = np.array(image_list)

        export_path = os.path.join(CONFIG["file-service"]["export"]["local_folder"], dataset_id)

        with h5py.File(os.path.join(export_path, filename), 'w') as dsetfile:
            dsetfile.create_dataset('export_data', data=json.dumps(export_data, ensure_ascii=False, default=default_json_converter), dtype=dt)
            dsetfile.create_dataset('examples', data=example_list, compression="gzip", compression_opts=9)
            dsetfile.create_dataset('images', data=image_list)
            dsetfile.create_dataset('parms', data=json.dumps(parms, ensure_ascii=False, default=default_json_converter), dtype=dt)


    def export_dataset(self, filename=None):
        """
        Export dataset to folder adding only differences if a base export exists
        """

        if not filename:
            filename = self.id + ".dset"

        self.load_all_images()

        export_data = {
            "export_version": "2",
            "export_date": pytz.utc.localize(datetime.datetime.now()),
            "dataset": self.dataset_name,
            "dataset_id": self.id,
            "num_examples": len(self.examples),
            "images_list": list(self.images.keys())
        }

        # Dataset.export_to_file(filename)
        Dataset.export_to_hdf5(export_data, self.parms, self.examples, self.images, filename, self.id)
        # if self._app_token is not None:

        log.info(f"Dataset exported to file {len(self.examples)} examples")


    @staticmethod
    def import_from_hdf5(filename, dataset_id, retrieve_images=True):
        """
        Import dataset from file in protobuf format
        """
        def convert_date(str_date):
            return dateutil.parser.isoparse(str_date).replace(tzinfo=None)

        export_path = os.path.join(CONFIG["file-service"]["export"]["local_folder"], dataset_id)
        with h5py.File(os.path.join(export_path, filename), 'r') as dsetfile:
            export_data = json.loads(dsetfile['export_data'][()])

            example_list = dsetfile['examples'][()].tolist()
            image_list = dsetfile['images'][()].tolist()
            parms = dsetfile['parms'][()]


        parms = json.loads(parms)
        parms["_id"] = ObjectId(dataset_id)
        parms["info"]["creation_date"] = convert_date(parms["info"].get("creation_date"))
        parms["info"]["modified_date"] = convert_date(parms["info"].get("modified_date"))

        examples = []
        for exp in example_list:
            exp = json.loads(exp)
            exp["_id"] = ObjectId()
            exp["date"] = convert_date(exp.get("date"))
            exp["dataset_id"] = ObjectId(dataset_id)
            if "modified_date" in exp:
                exp["modified_date"] = convert_date(exp.get("modified_date"))

            examples.append(exp)

        if not retrieve_images:
            return parms, examples

        images = {}
        for exp, img in zip(examples, image_list):
            images[exp["example"]] = img

        return export_data, parms, examples, images


    def import_dataset(self, filename=None):
        """
        Import dataset from folder with base and diffs
        """

        if not filename:
            if self.id:
                filename = self.id + ".dset"
            elif self.dataset_name:
                filename = self.dataset_name + ".dset"
            else:
                raise Exception(f'Must define import filename')

        export_data, self.parms, self.examples, self.images = Dataset.import_from_hdf5(filename, self.id)

        self.dataset_name = str(self.parms["name"])
        self.id = str(self.parms["_id"])

        if len(self.examples) != export_data["num_examples"]:
            raise Exception(f"Invalid number of examples. Imported: {len(self.examples)} - Exported: {export_data['num_examples']}")

        if len(self.images) != len(export_data["images_list"]):
            raise Exception(f"Invalid number of images. Imported: {len(self.images)} - Exported: {len(export_data['images_list'])}")

        log.info("Dataset imported from file %d examples" % len(self.examples))
# ---------------------------------------------------------------------------------------------------------------------------------
