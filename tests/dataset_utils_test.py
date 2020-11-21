import os
import sys

if __name__ == "__main__" and (__package__ is None or __package__ == ''):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eyeflow_sdk.log_obj import CONFIG
from eyeflow_sdk.dataset_utils import Dataset
#----------------------------------------------------------------------------------------------------------------------------------

db_config = CONFIG["db-service"]
cloud_parms = CONFIG["cloud"]

dset1 = Dataset("area_una_externa", db_config=db_config, cloud_parms=cloud_parms)
dset1.load_data()
dset1.export_dataset()

dset2 = Dataset("5f23184163f83e2fea546acd", db_config=db_config, cloud_parms=cloud_parms)
dset2.import_dataset()

assert(dset1.parms == dset2.parms)
assert(dset1.examples == dset2.examples)