rm -f dist/*
python3 inc_version.py
python3 setup.py sdist bdist_wheel
python3 -m twine upload --repository pypi dist/*
