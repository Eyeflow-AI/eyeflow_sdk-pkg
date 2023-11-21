"""
SiliconLife Eyeflow
Component to Identify and cut a Region of Interest in a image using a object_location model

Author: Alex Sobral de Freitas
"""

import os
import json
import jwt
import datetime

from pymongo.mongo_client import MongoClient
from bson.objectid import ObjectId
# ---------------------------------------------------------------------------------------------------------------------------------

def get_plat_token(platform, environment_name, exp_minutes=600):
    if platform == "dev":
        cred_file = os.path.join(os.environ['HOME'], ".eyeflow", "env_credentials_dev.json")
    elif platform == "beta":
        cred_file = os.path.join(os.environ['HOME'], ".eyeflow", "env_credentials_beta.json")
    elif platform == "prod":
        cred_file = os.path.join(os.environ['HOME'], ".eyeflow", "env_credentials_prod.json")
    else:
        raise Exception(f"Invalid platform: {platform}")

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
        "app_name": "generic_job",
        "environment_id": str(src_credentials["_id"]),
        "environment_name": environment_name,

        "endpoint": credentials["endpoint"],
        "job_parms": {
        },
        "bill_parms": credentials["bill_parms"],
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=exp_minutes)
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

    db_config = src_credentials["db_resource"]
    if "-pri" in db_config["db_url"]:
        idx = db_config["db_url"].index("-pri")
        db_config["db_url"] = db_config["db_url"][:idx] + db_config["db_url"][idx + 4:]

    db_src = MongoClient(db_config["db_url"])[db_config["db_name"]]

    return job_token, src_credentials, db_src
# ---------------------------------------------------------------------------------------------------------------------------------
