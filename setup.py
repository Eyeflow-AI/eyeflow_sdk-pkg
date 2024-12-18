import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eyeflow_sdk",
    version="1.2.9",
    author="Eyeflow.AI",
    author_email="support@eyeflow.ai",
    description="Functions and classes for development of Eyeflow Applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/siliconlife-ai/eyeflow_sdk-pkg",
    packages=setuptools.find_packages(),
    package_data={
        "": ["*.json", "*.yaml"],
    },
    install_requires=[
        "python-dateutil",
        "pymongo",
        "dnspython",
        "pika",
        "numpy",
        "opencv-python",
        "pillow",
        "protobuf",
        "h5py",
        "arrow",
        "psutil",
        "pynvml",
        "xmltodict",
        "pyyaml",
        "pyjwt",
        "azure-storage-blob"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
