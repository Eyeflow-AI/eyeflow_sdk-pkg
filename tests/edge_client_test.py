import sys
import pytz
from ..eyeflow_sdk.edge_client import *
from ..eyeflow_sdk.log_obj import log, CONFIG
import jwt
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
    dataset_id = "5e46bb04c2414cbcbb4fdb2e"
    if os.path.isfile(os.path.join(CONFIG["file-service"]["model"]["local_folder"], dataset_id, dataset_id + ".json")):
        os.remove(os.path.join(CONFIG["file-service"]["model"]["local_folder"], dataset_id, dataset_id + ".json"))

    res = get_model(app_token, dataset_id, CONFIG["file-service"]["model"]["local_folder"])
    assert(os.path.isfile(os.path.join(CONFIG["file-service"]["model"]["local_folder"], dataset_id, dataset_id + ".json")))
    assert(res == dataset_id)
    dataset_id = "6f23184163f83e2fea546acd"
    res = get_model(app_token, dataset_id, CONFIG["file-service"]["model"]["local_folder"])
    assert(res is None)


def test_upload_model(app_token):
    log.info("Testing upload_model")
    dataset_id = "5f2317bea6dacd7984e5baf5"
    model_info = {
        "_id": {"$oid": dataset_id},
        "size": 0,
        "train_parms": {"dataset": "parms"},
        "train_id": {"$oid": "60c1031760005700126234a6"},
        "train_date": {"$date": pytz.utc.localize(datetime.datetime.now())}
    }

    ckpt = "/opt/eyeflow/data/train/5f2317bea6dacd7984e5baf5/2021-09-23T15-56/check_point/"
    train_id = "614ca39fd81d2a6ca94fcfdb"
    train_info = {
        "size": 0,
        "train_parms": {"dataset": "parms"},
        "train_id": {"$oid": train_id},
        "train_history": [],
        "train_date": {"$date": pytz.utc.localize(datetime.datetime.now())}
    }

    if not os.path.isfile(os.path.join(CONFIG["file-service"]["model"]["local_folder"], dataset_id, dataset_id + ".json")):
        raise Exception(f"Model not found {dataset_id}")

    if not os.path.isfile(os.path.join(ckpt, dataset_id + ".json")):
        raise Exception(f"Hist not found {ckpt}")

    upload_model(
        app_token,
        dataset_id,
        model_info,
        CONFIG["file-service"]["model"]["local_folder"],
        train_id,
        train_info,
        ckpt
    )


def test_get_train(app_token):
    log.info("Testing get_train")
    dataset_id = "5f2317bea6dacd7984e5baf5"
    train_id = "614ca39fd81d2a6ca94fcfdb"
    if os.path.isfile(os.path.join(CONFIG["file-service"]["model-hist"]["local_folder"], dataset_id, dataset_id + ".json")):
        os.remove(os.path.join(CONFIG["file-service"]["model-hist"]["local_folder"], dataset_id, dataset_id + ".json"))

    res = get_train(app_token, dataset_id, train_id, CONFIG["file-service"]["model-hist"]["local_folder"])
    assert(os.path.isfile(os.path.join(CONFIG["file-service"]["model-hist"]["local_folder"], dataset_id, train_id, dataset_id + ".json")))
    assert(res == train_id)
    dataset_id = "6f23184163f83e2fea546acd"
    res = get_train(app_token, dataset_id, train_id, CONFIG["file-service"]["model-hist"]["local_folder"])
    assert(res is None)


def test_insert_train_event(app_token):
    log.info("Testing insert_train_event")
    dataset_id = "5f2317bea6dacd7984e5baf5"
    train_id = "614ca39fd81d2a6ca94fcfdb"
    event = {
        "train_id": {"$oid": train_id},
        "dataset": {"$oid": dataset_id},
        "dataset_name": "dataset_name",
        "dataset_type": "object_detection",
        "time": {"$date": pytz.utc.localize(datetime.datetime.now())},
        "event": "train_start",
        "train_parms": {"parms": "train_parms"},
        "total_epochs": 10,
        "steps_per_epoch": 100
    }
    res = insert_train_event(app_token, event)
    assert(res)


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
    app_token = sys.args[1]

    test_get_dataset(app_token)
    test_get_flow(app_token)
    test_get_model(app_token)
    test_get_flow_component(app_token)
    test_get_model_component(app_token)
    test_upload_extract(app_token)
    test_get_video(app_token)
    test_upload_video(app_token)
    test_upload_model(app_token)
    test_get_train(app_token)
    test_insert_train_event(app_token)
