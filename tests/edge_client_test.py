import os
import sys
import json
import datetime
from pymongo.mongo_client import MongoClient
from bson.objectid import ObjectId

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../eyeflow_sdk"))

from edge_client import *
from log_obj import log, CONFIG
import jwt
# ---------------------------------------------------------------------------------------------------------------------------------

def get_token(environment_name):
    cred_file = os.path.join(os.environ['HOME'], ".eyeflow", "env_credentials_dev.json")
    # cred_file = os.path.join(os.environ['HOME'], ".eyeflow", "env_credentials_beta.json")
    # cred_file = os.path.join(os.environ['HOME'], ".eyeflow", "env_credentials_prod.json")
    with open(cred_file) as fp:
        credentials = json.load(fp)

    db_auth_client = MongoClient(credentials["atlas"]["db_url"])
    db_auth = db_auth_client["eyeflow-auth"]

    src_credentials = db_auth.environment.find_one({"name": environment_name})
    if not src_credentials:
        raise Exception(f"Environment not found {environment_name}")

    job_id = ObjectId()
    token_id = ObjectId()

    token_payload = {
        "token_id": str(token_id),
        "job_id": str(job_id),
        "app_id": "608b0ac5a40cba9bb25206f7",
        "app_name": "dataset_train_job",
        "environment_id": str(src_credentials["_id"]),
        "environment_name": environment_name,

        "endpoint": credentials["endpoint"],
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=600)
    }

    key_file = credentials["key_file"]
    with open(key_file, 'r') as fp:
        private_key = fp.read()

    job_token = jwt.encode(token_payload, private_key, algorithm='RS256').decode('utf-8')

    edge_token = {
        "_id": token_id,
        "active": True,
        "token": job_token,
        "payload": token_payload,
        "token_key": ObjectId(credentials["edge_token_key"]),
        "acl_type": "user",
        "info": {
            "creation_date": datetime.datetime.now(datetime.timezone.utc)
        }
    }

    db_auth.edge_tokens.insert_one(edge_token)

    # db_auth.edge_tokens.delete_one({"_id": token_id})

    db_config = src_credentials["db_resource"]
    if "-pri" in db_config["db_url"]:
        idx = db_config["db_url"].index("-pri")
        db_config["db_url"] = db_config["db_url"][:idx] + db_config["db_url"][idx + 4:]

    db_src = MongoClient(db_config["db_url"])[db_config["db_name"]]

    return job_token, src_credentials, db_src
# ---------------------------------------------------------------------------------------------------------------------------------


def test_get_flow(app_token):
    log.info("Testing get_flow")
    parms = jwt.decode(app_token, options={"verify_signature": False})
    flow = get_flow(app_token, parms["job_parms"]["flow_id"])
    assert(flow["_id"] == parms["job_parms"]["flow_id"])
    flow_id = "70211eac368c7c00285fda57"
    flow = get_flow(app_token, flow_id)
    assert(flow is None)


def test_get_dataset(app_token):
    log.info("Testing get_dataset")
    dataset_id = "6f23184163f83e2fea546aff"
    dataset = get_dataset(app_token, dataset_id)
    assert(dataset is None)
    dataset_id = "5f23184163f83e2fea546acd"
    dataset = get_dataset(app_token, dataset_id)
    assert(dataset["dataset_parms"]["_id"] == dataset_id)


def test_get_model(app_token):
    log.info("Testing get_model")
    dataset_id = "5d420731ddfafbbbdeb17277"
    if os.path.isfile(os.path.join(CONFIG["file-service"]["model"]["local_folder"], dataset_id + ".json")):
        os.remove(os.path.join(CONFIG["file-service"]["model"]["local_folder"], dataset_id + ".json"))
    if os.path.isfile(os.path.join(CONFIG["file-service"]["model"]["local_folder"], dataset_id, dataset_id + ".json")):
        os.remove(os.path.join(CONFIG["file-service"]["model"]["local_folder"], dataset_id, dataset_id + ".json"))

    res = get_model(app_token, dataset_id, CONFIG["file-service"]["model"]["local_folder"])
    assert(os.path.isfile(os.path.join(CONFIG["file-service"]["model"]["local_folder"], dataset_id, dataset_id + ".json")))
    assert(res["dataset_id"] == dataset_id)
    dataset_id = "6f23184163f83e2fea546acd"
    res = get_model(app_token, dataset_id, CONFIG["file-service"]["model"]["local_folder"])
    assert(res["train_id"] is None)


def test_get_flow_component(app_token):
    log.info("Testing get_flow_component")
    flow_component_id = "5d96674dc00beadd67156657"
    if os.path.isfile(os.path.join(CONFIG["file-service"]["flow-components"]["local_folder"], flow_component_id, "roi_cutter.py")):
        os.remove(os.path.join(CONFIG["file-service"]["flow-components"]["local_folder"], flow_component_id, "roi_cutter.py"))

    get_flow_component(app_token, flow_component_id, CONFIG["file-service"]["flow-components"]["local_folder"])
    assert(os.path.isfile(os.path.join(CONFIG["file-service"]["flow-components"]["local_folder"], flow_component_id, "roi_cutter.py")))


def test_get_model_component(app_token):
    log.info("Testing get_model_component")
    model_component_id = "6143a1faef5cc63fd4c177b1"
    if os.path.isfile(os.path.join(CONFIG["file-service"]["model-components"]["local_folder"], model_component_id, "objdet_af.py")):
        os.remove(os.path.join(CONFIG["file-service"]["model-components"]["local_folder"], model_component_id, "objdet_af.py"))

    get_model_component(app_token, model_component_id, CONFIG["file-service"]["model-components"]["local_folder"])
    assert(os.path.isfile(os.path.join(CONFIG["file-service"]["model-components"]["local_folder"], model_component_id, "objdet_af.py")))


def test_upload_extract(app_token):
    log.info("Testing upload_extract")
    dataset_id = "6141ea4579832900192fd211"
    upload_extract(app_token, dataset_id, CONFIG["file-service"]["extract"]["local_folder"])


def test_get_video(app_token):
    log.info("Testing get_video")
    video_id = "60211f2c58a6bd00c1715529"
    if os.path.isfile(os.path.join(CONFIG["file-service"]["video"]["local_folder"], video_id + ".mp4")):
        os.remove(os.path.join(CONFIG["file-service"]["video"]["local_folder"], video_id + ".mp4"))

    get_video(app_token, video_id, CONFIG["file-service"]["video"]["local_folder"])
    assert(os.path.isfile(os.path.join(CONFIG["file-service"]["video"]["local_folder"], video_id + ".mp4")))


def test_upload_video(app_token):
    log.info("Testing upload_video")
    video_id = "611facc8d1fe49001969cd33"
    annotation = {
        "start_date": {
            "$date": "2021-09-08T18:03:54.833Z"
        },
        "video_process_id": {
            "$oid": "60c1031760005700126234a6"
        },
        "flow_id": "60211eac368c7c00285fda57",
        "init_time": 0,
        "end_time": 66,
        "original_video": "611facc8d1fe49001969cd33.mp4",
        "init_frame": 0,
        "end_frame": 1980,
        "annotated_video": "611facc8d1fe49001969cd33_annotated.mp4",
        "annotated_data": "611facc8d1fe49001969cd33_output.json",
        "end_date": {
            "$date": "2021-09-08T18:05:33.845Z"
        }
    }

    video_file = "611facc8d1fe49001969cd33_annotated.mp4"
    output_file = "611facc8d1fe49001969cd33_output.json"
    video_folder = os.path.join(CONFIG["file-service"]["video"]["local_folder"], "output")
    if not os.path.isfile(os.path.join(video_folder, video_file)):
        raise Exception(f"File not found {video_file}")

    upload_video(app_token, video_id, annotation, video_file, output_file, video_folder)



if __name__ == "__main__":
    job_token, src_credentials, db_src = get_token("SiliconLife.AI")

    test_get_dataset(job_token)
    test_get_flow(job_token)
    test_get_flow_component(job_token)
    test_get_model_component(job_token)
    test_upload_extract(job_token)
    test_get_video(job_token)
    test_upload_video(job_token)
    test_get_model(job_token)
