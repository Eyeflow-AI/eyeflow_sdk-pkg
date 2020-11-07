
from eyeflow_sdk.cloud_store import azure
from eyeflow_sdk.log_obj import CONFIG

def test():
    folder_name = "export"
    resource_id = None

    _azure = azure.Connector(**CONFIG["cloud"])

    #List Files Test
    files_info = _azure.list_files_info(folder_name, resource_id=resource_id)
    print(f'Files: {files_info}')

    files = _azure.list_files(folder_name, resource_id=resource_id)
    print(f'Files: {files}')

    #Download Blob Test
    print(f'Downloading: {files[0]}')
    data = _azure.download_file(
        folder=folder_name,
        resource_id=resource_id,
        filename=files[0]
        )

    print(f'Uploading: test.jpg')
    _azure.upload_file(
        folder=folder_name,
        resource_id=resource_id,
        filename='test.jpg',
        data=data
        )

    print("Get File info")
    info = _azure.get_file_info(
        folder=folder_name,
        resource_id=resource_id,
        filename='test.jpg'
        )
    print(f"File_info: {info}")

    #Download uploaded data
    print(f'Downloading: test.jpg')
    data = _azure.download_file(
        folder=folder_name,
        resource_id=resource_id,
        filename='test.jpg'
        )

    #Delete uploaded data
    print(f'Deleting: test.jpg')
    _azure.delete_file(
        folder=folder_name,
        resource_id=resource_id,
        filename='test.jpg'
        )

    print(f'Writing test.jpg to /tmp/test.jpg')
    with open('/tmp/test.jpg', "wb") as my_blob:
        my_blob.write(data)
#----------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    test()
