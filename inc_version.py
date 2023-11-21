import os
import re

version_file = os.path.join(os.path.dirname(__file__), "setup.py")
version_pat = re.compile(r'(\s*version\s*=\s*)"(.*?)"')

with open(version_file, 'r') as fp:
    text = fp.readlines()

with open(version_file, 'w') as fp:
    for lin in text:
        version_line = version_pat.match(lin)
        if version_line is not None:
            old_version = version_line[2]
            old_version = old_version.replace('"', '')
            inc_version = [int(v) for v in old_version.split('.')]
            version = f"{inc_version[0]}.{inc_version[1]}.{inc_version[2] + 1}"
            new_version = f'{version_line[1]}"{version}",\n'
            fp.write(new_version)
        else:
            fp.write(lin)
