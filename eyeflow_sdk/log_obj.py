"""
SiliconLife Eyeflow
Log singleton object

Author: Alex Sobral de Freitas
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import yaml
#----------------------------------------------------------------------------------------------------------------------------------

if "CONF_PATH" not in os.environ:
    os.environ["CONF_PATH"] = "/opt/eyeflow/conf"

with open(os.path.join(os.environ["CONF_PATH"], "eyeflow_conf.yaml"), "r") as ymlfile:
    CONFIG = yaml.safe_load(ymlfile)
#----------------------------------------------------------------------------------------------------------------------------------

class LogObj:
    """
    Singletron class to operate log
    """

    class __LogObj:
        def __init__(self):
            """
            Initialize log
            """
            self.application_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)

            log_path = Path(CONFIG["log"]["log_folder"])

            if not log_path.is_dir():
                log_path.mkdir(parents=True, exist_ok=True)

            log_file = Path(log_path, f"{self.application_name}.log")
            print("Log File: %s" % str(log_file))

            try:
                log_file_handler = RotatingFileHandler(
                    filename=str(log_file),
                    mode='a',
                    maxBytes=(10 * 1024 * 1024),
                    backupCount=5
                    )
            except:
                raise IOError("Couldn't create/open file \"" + str(log_file) + "\". Check permissions.")

            log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s'))
            log_file_handler.setLevel(logging.INFO)
            self.logger.addHandler(log_file_handler)

            # create console handler and set level to debug
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            console_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(console_handler)

        def __getattr__(self, name):
            return getattr(self.logger, name)

    instance = None

    def __init__(self):
        if not LogObj.instance:
            LogObj.instance = LogObj.__LogObj()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __str__(self):
        return 'Log singletron object'
#----------------------------------------------------------------------------------------------------------------------------------

log = LogObj()
#----------------------------------------------------------------------------------------------------------------------------------
