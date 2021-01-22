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
    dst_credentials_silicon = {
        "db-service": {
            "db_url": "mongodb+srv://app_user-dev:JcP9vJQgzpTKJvad@eyeflow-dev.9bm6s.mongodb.net/5fcd69ce139188003f9eaa83?retryWrites=true&w=majority",
            "db_name": "5fcd69ce139188003f9eaa83"
        },
        "cloud": {
            "provider": "azure",
            "account_url": "https://5fcd69ce139188003f9eaa83.blob.core.windows.net",
            "account_key": "owbO3pscQ4B8LifggwSTLOUETl2hF8i6pBlzq8PMdxc/Zk3uVIEpZBhVMHFhShtxlegQcz8V+pNfALbcpUDm3Q==",
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=5fcd69ce139188003f9eaa83;AccountKey=owbO3pscQ4B8LifggwSTLOUETl2hF8i6pBlzq8PMdxc/Zk3uVIEpZBhVMHFhShtxlegQcz8V+pNfALbcpUDm3Q==;EndpointSuffix=core.windows.net"
        }
    }

    dataset = Dataset(dataset_id, db_config=dst_credentials_silicon["db-service"], cloud_parms=dst_credentials_silicon["cloud"])
    dataset.load_data()
    dataset.update_default_parms()
    component = dataset.parms["network_parms"]["dnn_parms"]["component"]
#----------------------------------------------------------------------------------------------------------------------------------

test_default_parms()
