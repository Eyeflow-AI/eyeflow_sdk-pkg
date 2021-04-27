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
import dateutil.parser

import unicodedata
import re

import cv2
import numpy as np
import h5py

from pymongo import MongoClient
from bson.objectid import ObjectId

from eyeflow_sdk.log_obj import log, CONFIG
from eyeflow_sdk.file_access import FileAccess
#----------------------------------------------------------------------------------------------------------------------------------

def slugify(value):
    """
    Normalizes string, removes non-alpha characters, and converts spaces to underscore.
    """
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode()
    value = str(re.sub(r'[^\w\s-]', '', value).strip())
    value = str(re.sub(r'[\s]+', '_', value))
    return value
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
    def __init__(self, dataset, db_config=None, cloud_parms=None):
        if dataset != slugify(dataset):
            raise Exception(f"Invalid dataset name {dataset} - Sugestion: {slugify(dataset)}")

        self.id = None
        self.dataset_name = None

        if len(dataset) == 24:
            try:
                self.id = str(ObjectId(dataset))
            except:
                self.dataset_name = dataset
                self.id = None
        else:
            self.dataset_name = dataset

        if db_config is not None:
            self.db_config = db_config
        else:
            self.db_config = CONFIG["db-service"]

        self._cloud_parms = cloud_parms
        self._file_ac = FileAccess(
            storage="dataset",
            resource_id=self.id,
            cloud_parms=self._cloud_parms
        )

        self.parms = {}
        self.examples = []
        self.images = {}


    @staticmethod
    def get_mongo_database(db_config):
        """
        Returns a client to mongo_db access
        """
        client = MongoClient(db_config["db_url"])
        return client[db_config["db_name"]]


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
    def get_network_default_parms():
        """
        Returns default network parms
        """
        filename = os.path.join(os.path.dirname(__file__), 'network_default_parms.json')
        with open(filename, 'r', newline='', encoding='utf8') as fp:
            default_parms = json.load(fp)

        return default_parms


    @staticmethod
    def get_data_augmentation_default_parms():
        """
        Returns default data augmentation parms
        """
        filename = os.path.join(os.path.dirname(__file__), 'data_augmentation_default_parms.json')
        with open(filename, 'r', newline='', encoding='utf8') as fp:
            default_parms = json.load(fp)

        return default_parms


    @staticmethod
    def get_dataset_types():
        """
        Returns dataset types
        """
        filename = os.path.join(os.path.dirname(__file__), 'dataset_types.json')
        with open(filename, 'r', newline='', encoding='utf8') as fp:
            default_parms = json.load(fp)

        return default_parms


    def update_default_parms(self):
        """
        Performs an update in parms with all default parms read from database
        """
        network_parms = {}
        dataset_network_parms = self.parms.get("network_parms", network_parms)
        network_default_parms = Dataset.get_network_default_parms()
        if network_default_parms:
            network_parms.update(network_default_parms["network_parms"][self.parms["info"]["type"]])

        dataset_default_parms = Dataset.get_dataset_default_parms()
        if dataset_default_parms:
            network_parms.update(dataset_default_parms["network_parms"][self.parms["info"]["type"]])

        network_parms.update(dataset_network_parms)
        self.parms.update({"network_parms": network_parms})

        data_augmentation_default_parms = Dataset.get_data_augmentation_default_parms()
        if data_augmentation_default_parms:
            data_augmentation_default_parms = data_augmentation_default_parms["default_parms"]
            data_augmentation_default_parms.update(self.parms.get("data_augmentation_parms", {}))
            self.parms.update({"data_augmentation_parms": data_augmentation_default_parms})


    def load_data(self):
        """
        Load dataset data from database
        """
        db_mongo = Dataset.get_mongo_database(self.db_config)

        if self.id:
            self.parms = db_mongo.dataset.find_one({"_id": ObjectId(self.id)})

            if self.parms is None:
                raise Exception(f'Dataset not found: {self.id}')

            self.dataset_name = str(self.parms["name"])
        else:
            self.parms = db_mongo.dataset.find_one({"name": self.dataset_name})

            if self.parms is None:
                raise Exception(f'Dataset not found: {self.dataset_name}')

            self.id = str(self.parms["_id"])

        cursor = db_mongo.example.find({
            "dataset_id": self.parms["_id"],
            "active": {"$ne": False}
        })
        if cursor is not None:
            for exp in cursor:
                self.examples.append(exp)

        log.info(f"Load dataset from database {len(self.examples)} examples")


    def load_images_from_disk(self, origin="cloud"):
        """
        Load images from disk
        """
        if self._cloud_parms is not None:
            self._file_ac.sync_files(origin=origin)

        for exp in self.examples:
            with self._file_ac.open(exp["example"], 'rb') as fp:
                self.images[exp["example"]] = fp.read()


    def load_all_images(self):
        """
        Load images in memory
        """
        for exp in self.examples:
            if exp["example"] in self.images:
                continue

            if self._file_ac.is_file(exp["example"]):
                with self._file_ac.open(exp["example"], 'rb') as fp:
                    self.images[exp["example"]] = fp.read()
            else:
                if self._cloud_parms is None:
                    raise Exception('Image not found and cloud parms not set')

                self.images[exp["example"]] = self._file_ac.load_cloud_file(exp["example"])


    def save_data_db(self):
        """
        Save dataset data to database
        """
        db_mongo = Dataset.get_mongo_database(self.db_config)

        dataset_collection = db_mongo.dataset
        self.parms["_id"] = ObjectId(self.id)
        self.parms["info"]["package"] = ObjectId(self.parms["info"]["package"])
        self.parms["info"]['creation_date'] = datetime.datetime.now()
        self.parms["info"]['modified_date'] = datetime.datetime.now()

        dataset_collection.delete_many({"_id": self.parms['_id']})
        dataset_collection.insert_one(self.parms)

        example_collection = db_mongo.example
        # example_collection.delete_many({"dataset": self.parms['name']})
        example_collection.delete_many({"dataset_id": self.parms['_id']})

        for exp in self.examples:
            exp["dataset"] = self.parms['name']
            exp["dataset_id"] = self.parms['_id']
            example_collection.insert_one(exp)

        self.save_images_to_disk()

        log.info("Inserted dataset into database %d examples" % len(self.examples))


    def save_images_to_disk(self, origin="local"):
        """
        Save images to disk
        """
        self.load_all_images()

        for exp in self.examples:
            if self._file_ac.is_file(exp["example"]):
                self._file_ac.remove_file(exp["example"])

            with self._file_ac.open(exp["example"], 'wb') as fp:
                fp.write(self.images[exp["example"]])

        if self._cloud_parms is not None:
            self._file_ac.sync_files(origin=origin)


    def get_example_img(self, example_img):
        """
        Returns an image in opencv format
        """
        def load_file_from_cloud(filename):
            if self._cloud_parms is None:
                raise Exception('Cloud parms not set')

            return self._file_ac.load_cloud_file(filename)


        if example_img not in self.images:
            self.images[example_img] = load_file_from_cloud(example_img)

        # by default all images are in RGB
        return cv2.cvtColor(cv2.imdecode(np.frombuffer(self.images[example_img], dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


    def get_train_subsets(self, shuffle=True):
        """
        Partitioning of dataset in 3 groups: train, validation and test
        """
        if not self.parms:
            raise Exception('Must load or import dataset first')

        train_dataset = Dataset(self.id, db_config=self.db_config, cloud_parms=self._cloud_parms)
        train_dataset.parms = copy.deepcopy(self.parms)

        val_dataset = Dataset(self.id, db_config=self.db_config, cloud_parms=self._cloud_parms)
        val_dataset.parms = copy.deepcopy(self.parms)

        test_dataset = Dataset(self.id, db_config=self.db_config, cloud_parms=self._cloud_parms)
        test_dataset.parms = copy.deepcopy(self.parms)

        if shuffle:
            random.shuffle(self.examples)

        val_size = int(len(self.examples) * self.parms["network_parms"]["val_size"])
        if val_size < 1:
            val_size = 1

        test_size = int(len(self.examples) * self.parms["network_parms"]["test_size"])
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

        file_ac = FileAccess(storage="export", resource_id=dataset_id)
        export_path = file_ac.get_local_folder()
        with h5py.File(os.path.join(export_path, filename), 'w') as dsetfile:
            dsetfile.create_dataset('export_data', data=json.dumps(export_data, ensure_ascii=False, default=default_json_converter), dtype=dt)
            dsetfile.create_dataset('examples', data=example_list, compression="gzip", compression_opts=9)
            dsetfile.create_dataset('images', data=image_list)
            dsetfile.create_dataset('parms', data=json.dumps(parms, ensure_ascii=False, default=default_json_converter), dtype=dt)


    def export_dataset(self, filename=None):
        """
        Export dataset to folder adding only differences if a base export exists
        """

        file_ac = FileAccess(storage="export", resource_id=self.id, cloud_parms=self._cloud_parms)
        if not filename:
            filename = self.id + ".dset"

        self.load_all_images()

        export_data = {
            "export_version": "2",
            "export_date": datetime.datetime.now(),
            "dataset": self.dataset_name,
            "dataset_id": self.id,
            "num_examples": len(self.examples),
            "images_list": list(self.images.keys())
        }

        # Dataset.export_to_file(filename)
        Dataset.export_to_hdf5(export_data, self.parms, self.examples, self.images, filename, self.id)
        if self._cloud_parms is not None:
            file_ac.sync_files(
                file_list=[filename],
                origin="local"
            )
        log.info(f"Dataset exported to file {len(self.examples)} examples")


    @staticmethod
    def import_from_hdf5(filename, dataset_id, retrieve_images=True):
        """
        Import dataset from file in protobuf format
        """
        def convert_date(str_date):
            return dateutil.parser.isoparse(str_date).replace(tzinfo=None)

        file_ac = FileAccess(storage="export", resource_id=dataset_id)
        export_path = file_ac.get_local_folder()
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

        file_ac = FileAccess(storage="export", resource_id=self.id, cloud_parms=self._cloud_parms)

        if self._cloud_parms is not None:
            file_ac.sync_files(file_list=[filename], origin="cloud")

        if not file_ac.is_file(filename):
            raise Exception(f'Export file {filename} does not exist')

        export_data, self.parms, self.examples, self.images = Dataset.import_from_hdf5(filename, self.id)

        self.dataset_name = str(self.parms["name"])
        self.id = str(self.parms["_id"])

        if len(self.examples) != export_data["num_examples"]:
            raise Exception(f"Invalid number of examples. Imported: {len(self.examples)} - Exported: {export_data['num_examples']}")

        if len(self.images) != len(export_data["images_list"]):
            raise Exception(f"Invalid number of images. Imported: {len(self.images)} - Exported: {len(export_data['images_list'])}")

        log.info("Dataset imported from file %d examples" % len(self.examples))
# ---------------------------------------------------------------------------------------------------------------------------------
