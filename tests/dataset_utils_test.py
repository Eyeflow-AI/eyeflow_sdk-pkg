import os
import sys
import datetime

if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'eyeflow_sdk'))

from log_obj import CONFIG
from dataset_utils import Dataset
#----------------------------------------------------------------------------------------------------------------------------------

def test_export():
    export_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    script_parms = {
        "dataset_id": "5f2317bea6dacd7984e5baf5",
        "filename": f"area_treta_externa-{export_date}.dset"
    }

    dset1 = Dataset(script_parms["dataset_id"], db_config=CONFIG["db-service"], cloud_parms=CONFIG["cloud"])
    dset1.load_data()
    dset1.export_dataset(script_parms["filename"])

    dset2 = Dataset(script_parms["dataset_id"], db_config=CONFIG["db-service"], cloud_parms=CONFIG["cloud"])
    dset2.import_dataset(script_parms["filename"])
    dset2.save_data_db()

    assert(dset1.parms["_id"] == dset2.parms["_id"])
    assert(dset1.parms["classes"] == dset2.parms["classes"])
    assert(dset1.parms["network_parms"] == dset2.parms["network_parms"])
    assert(dset1.examples == dset2.examples)
#----------------------------------------------------------------------------------------------------------------------------------

def test_default_parms():
    dataset_id = "600a260286aa5b0058397154"

    dataset = Dataset(dataset_id, db_config=CONFIG["db-service"], cloud_parms=CONFIG["cloud"])
    dataset.load_data()
    dataset.update_default_parms()
    component = dataset.parms["network_parms"]["dnn_parms"]["component"]
#----------------------------------------------------------------------------------------------------------------------------------

test_export()
# test_default_parms()
