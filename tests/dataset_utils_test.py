import os
import sys

if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eyeflow_sdk'))

from log_obj import CONFIG
from dataset_utils import Dataset
#----------------------------------------------------------------------------------------------------------------------------------

def test_export():
    db_config = CONFIG["db-service"]
    cloud_parms = CONFIG["cloud"]

    dset1 = Dataset("area_una_externa", db_config=db_config, cloud_parms=cloud_parms)
    dset1.load_data()
    dset1.export_dataset()

    dset2 = Dataset("5f23184163f83e2fea546acd", db_config=db_config, cloud_parms=cloud_parms)
    dset2.import_dataset()

    assert(dset1.parms == dset2.parms)
    assert(dset1.examples == dset2.examples)
#----------------------------------------------------------------------------------------------------------------------------------

def test_default_parms():
    dataset_id = "600a260286aa5b0058397154"

    dataset = Dataset(dataset_id, db_config=CONFIG["db-service"], cloud_parms=CONFIG["cloud"])
    dataset.load_data()
    dataset.update_default_parms()
    component = dataset.parms["network_parms"]["dnn_parms"]["component"]
#----------------------------------------------------------------------------------------------------------------------------------

test_default_parms()
