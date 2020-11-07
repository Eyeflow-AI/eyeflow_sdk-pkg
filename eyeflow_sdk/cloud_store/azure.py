"""
SiliconLife Eyeflow
Class for access files synchronized with azure cloud storage

Author: Gabriel Melo
"""

# pip3 install azure-storage-blob
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
#----------------------------------------------------------------------------------------------------------------------------------

class Connector():
    """
    Azure blobs access
    """

    def __init__(self, **kwargs):
        self.account_url = kwargs["account_url"]
        self.account_key = kwargs["account_key"]
        self.client = BlobServiceClient(
            account_url=self.account_url,
            credential=self.account_key
            )


    def _get_container_client(self, folder):
        return self.client.get_container_client(container=folder)


    def _get_blob_client(self, folder, filename):
        return self.client.get_blob_client(container=folder, blob=filename)


    def list_files(self, folder, resource_id=None, prefix='', sufix='', full_path=False):
        container_client = self._get_container_client(folder)
        file_list = []
        for blob in container_client.list_blobs(name_starts_with=resource_id if resource_id is not None else ''):
            if full_path:
                blob_name = blob['name']
            else:
                blob_name = blob['name'].split('/').pop()

            if blob_name.startswith(prefix) and blob_name.endswith(sufix):
                file_list.append(blob_name)

        return file_list


    def list_files_info(self, folder, resource_id=None, prefix='', sufix='', full_path=False):
        """
        Returns a list with info of all files from folder
        [{"filename": string,
          "creation_date": timestamp,
          "modified_date": timestamp,
          "file_size": int}
        ]
        """
        container_client = self._get_container_client(folder)
        file_list = []
        for blob in container_client.list_blobs(name_starts_with=resource_id + '/' if resource_id is not None else ''):
            if full_path:
                blob_name = blob['name']
            elif resource_id is not None:
                blob_name = blob['name'].replace(resource_id + '/', '', 1)
            else:
                blob_name = blob['name']

            if blob_name.startswith(prefix) and blob_name.endswith(sufix):
                file_info = {"filename": blob_name,
                             "creation_date": blob["creation_time"],
                             "modified_date": blob["last_modified"],
                             "file_size": blob["size"]
                            }

                file_list.append(file_info)

        return file_list


    def is_file(self, folder, filename, resource_id=None):
        """
        Returns true if file exists
        """
        if resource_id is not None:
            blob_name = f"{resource_id}/{filename}"
        else:
            blob_name = filename

        try:
            blob_client = self._get_blob_client(folder, blob_name)
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False


    def get_file_info(self, folder, filename, resource_id=None):
        """
        Returns file info in a dict
        {"filename": string,
         "creation_date": timestamp,
         "modified_date": timestamp,
         "file_size": int
        }
        """
        if resource_id is not None:
            blob_name = f"{resource_id}/{filename}"
        else:
            blob_name = filename

        blob_client = self._get_blob_client(folder, blob_name)
        blob = blob_client.get_blob_properties()
        blob_name = blob['name'].split('/').pop()

        blob_info = {"filename": blob_name,
                     "creation_date": blob["creation_time"],
                     "modified_date": blob["last_modified"],
                     "file_size": blob["size"]
                    }

        return blob_info


    def download_file(self, folder, filename, resource_id=None):
        if resource_id is not None:
            blob_name = f"{resource_id}/{filename}"
        else:
            blob_name = filename

        blob_client = self._get_blob_client(folder, blob_name)
        try:
            download_stream = blob_client.download_blob()
        except Exception as err:
            raise Exception(f'Error downloading {filename}: {err}')
        else:
            data = download_stream.readall()
            return data


    def upload_file(self, folder, filename, data, resource_id=None):
        if not isinstance(data, bytes):
            raise Exception(f'Data should be byte type, {type(data)} detected.')

        if resource_id is not None:
            blob_name = f"{resource_id}/{filename}"
        else:
            blob_name = filename

        blob_client = self._get_blob_client(folder, blob_name)
        try:
            blob_client.upload_blob(data=data)
        except Exception as err:
            raise Exception(f'Error uploading {filename}: {err}')
        else:
            # print(f'Successfully uploaded {filename}')
            pass


    def delete_file(self, folder, filename, resource_id=None):
        if resource_id is not None:
            blob_name = f"{resource_id}/{filename}"
        else:
            blob_name = filename

        blob_client = self._get_blob_client(folder, blob_name)
        try:
            blob_client.delete_blob()
        except Exception as err:
            raise Exception(f'Error deleting {filename}: {err}')
        else:
            # print(f'Successfully deleted {blob_name}')
            pass
#----------------------------------------------------------------------------------------------------------------------------------
