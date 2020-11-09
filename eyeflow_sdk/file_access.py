"""
SiliconLife Eyeflow
Class for access files synchronized with cloud storage

Author: Alex Sobral de Freitas
"""

import os
from pathlib import Path
import importlib
import datetime
import time
import traceback

from eyeflow_sdk.log_obj import log, CONFIG
#----------------------------------------------------------------------------------------------------------------------------------

class FileAccess():
    """
    Class for handle files cloud synchronized transparently
    """
    def __init__(self, storage, resource_id=None, cloud_parms=None):
        config_parms = CONFIG["file-service"]

        if storage not in config_parms:
            raise Exception(f"Unknown storage: {storage}")

        self.storage = storage
        self.resource_id = resource_id

        if resource_id is not None:
            self._local_folder = config_parms[storage]["local_folder"] + "/" + resource_id
        else:
            self._local_folder = config_parms[storage]["local_folder"]

        self._cloud_folder = None
        if "cloud_folder" in config_parms[storage]:
            self._cloud_folder = config_parms[storage]["cloud_folder"]

        # make folder if not exists
        folder_path = Path(self._local_folder)
        if not folder_path.is_dir():
            folder_path.mkdir(parents=True, exist_ok=True)

        self._cloud_obj = None
        if cloud_parms is not None:
            comp_lib = importlib.import_module(f'eyeflow_sdk.cloud_store.{cloud_parms["provider"]}')
            self._cloud_obj = comp_lib.Connector(**cloud_parms)

    @staticmethod
    def _get_list_files_info(folder, prefix='', sufix=''):
        file_list = []
        for filename in os.listdir(folder):
            if os.path.isfile(os.path.join(folder, filename)) \
               and filename.startswith(prefix) and filename.endswith(sufix):
                file_stat = os.stat(os.path.join(folder, filename))
                file_info = {"filename": filename,
                             "creation_date": datetime.datetime.fromtimestamp(file_stat.st_mtime, datetime.timezone.utc),
                             "modified_date": datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(folder, filename)), datetime.timezone.utc),
                             "file_size": file_stat.st_size
                            }

                file_list.append(file_info)
            elif os.path.isdir(os.path.join(folder, filename)):
                subfolder = os.path.join(folder, filename)
                subfolder_files = FileAccess._get_list_files_info(subfolder, prefix, sufix)
                for finfo in subfolder_files:
                    finfo["filename"] = os.path.join(os.path.split(subfolder)[1], finfo["filename"])

                file_list.extend(subfolder_files)

        return file_list


    def sync_files(self, origin, file_list=None):
        """
        Synchronizes files between local and cloud folder
        If file_list then synchronize only files in this list
        """
        if self._cloud_obj is None:
            raise Exception("Cloud provider not defined")

        if self._cloud_folder is None:
            raise Exception("Cloud folder not defined")

        if origin not in ["cloud", "local", "both"]:
            raise Exception(f"'origin'={origin} must be one of: ('cloud', 'local', 'both')")

        try:
            local_files = self._get_list_files_info(self._local_folder)
            cloud_files = self._cloud_obj.list_files_info(folder=self._cloud_folder, resource_id=self.resource_id)

            if origin in ["cloud", "both"]:
                download_files = []
                for c_file in cloud_files:
                    if file_list is not None and c_file["filename"] not in file_list:
                        continue

                    for l_file in local_files:
                        if l_file["filename"] == c_file["filename"] and \
                        l_file["file_size"] == c_file["file_size"] and \
                        l_file["modified_date"] >= c_file["modified_date"]:
                            break
                    else:
                        download_files.append(c_file)

                if download_files:
                    log.info(f"Storage: {self.storage} - Resource-ID: {self.resource_id} - Download {len(download_files)} files")
                    for c_file in download_files:
                        file_data = self._cloud_obj.download_file(
                            folder=self._cloud_folder,
                            resource_id=self.resource_id,
                            filename=c_file["filename"]
                        )

                        file_path = os.path.join(self._local_folder, c_file["filename"])
                        dirname = os.path.dirname(file_path)
                        if not os.path.isdir(dirname):
                            os.makedirs(dirname, exist_ok=True)

                        with open(file_path, "wb") as fp:
                            fp.write(file_data)

            if origin in ["local", "both"]:
                upload_files = []
                for l_file in local_files:
                    if file_list is not None and l_file["filename"] not in file_list:
                        continue

                    for c_file in cloud_files:
                        if l_file["filename"] == c_file["filename"] and \
                        l_file["file_size"] == c_file["file_size"] and \
                        l_file["modified_date"] <= c_file["modified_date"]:
                            break
                    else:
                        upload_files.append(l_file)

                if upload_files:
                    log.info(f"Storage: {self.storage} - Resource-ID: {self.resource_id} - Upload {len(upload_files)} files")
                    for l_file in upload_files:
                        file_path = os.path.join(self._local_folder, l_file["filename"])
                        with open(file_path, "rb") as fp:
                            file_data = fp.read()

                        if self._cloud_obj.is_file(
                                folder=self._cloud_folder,
                                resource_id=self.resource_id,
                                filename=l_file["filename"],
                            ):
                            self._cloud_obj.delete_file(
                                folder=self._cloud_folder,
                                resource_id=self.resource_id,
                                filename=l_file["filename"],
                            )

                        file_data = self._cloud_obj.upload_file(
                            folder=self._cloud_folder,
                            resource_id=self.resource_id,
                            filename=l_file["filename"],
                            data=file_data
                        )

                        # Set local file modified time to now() to prevent download the same file after
                        date = datetime.datetime.now()
                        mod_time = time.mktime(date.timetuple())
                        try:
                            os.utime(file_path, (mod_time, mod_time))
                        except:
                            pass

        except Exception as excp:
            log.error(traceback.format_exc())
            log.error(f"sync_files error: {excp}")
            raise excp


    def purge_files(self, max_files=800, max_days=None):
        """
        Purge local and cloud files to max_files from de older to newer
        """
        local_files = self._get_list_files_info(self._local_folder)
        if len(local_files) > max_files:
            date_list = [(l_file["filename"], l_file["modified_date"]) for l_file in local_files]
            exclude_list = sorted(date_list, key=lambda x: x[1])[:len(local_files) - max_files]
            log.info(f"Purge local files: {len(exclude_list)}")
            for filename, _ in exclude_list:
                try:
                    os.remove(os.path.join(self._local_folder, filename))
                except:
                    pass

        if self._cloud_obj is None or self._cloud_folder is None:
            return

        cloud_files = self._cloud_obj.list_files_info(folder=self._cloud_folder, resource_id=self.resource_id)
        if len(cloud_files) > max_files:
            date_list = [(l_file["filename"], l_file["modified_date"]) for l_file in cloud_files]
            exclude_list = sorted(date_list, key=lambda x: x[1])[:len(cloud_files) - max_files]
            log.info(f"Purge cloud files: {len(exclude_list)}")
            for filename, _ in exclude_list:
                try:
                    self._cloud_obj.delete_file(
                        folder=self._cloud_folder,
                        resource_id=self.resource_id,
                        filename=filename,
                    )
                except:
                    pass


    def get_file_path(self, filename):
        return os.path.join(self._local_folder, filename)


    def get_local_folder(self):
        return self._local_folder


    def open(self, filename, *args, **kwargs):
        """
        Returns a file pointer to a local file
        """
        file_path = os.path.join(self._local_folder, filename)
        return open(file_path, *args, **kwargs)


    def is_file(self, filename):
        """
        Returns true if file exists
        """
        file_path = os.path.join(self._local_folder, filename)
        return os.path.isfile(file_path)


    def remove_file(self, filename):
        """
        Remove a local file
        """
        file_path = os.path.join(self._local_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
#----------------------------------------------------------------------------------------------------------------------------------
