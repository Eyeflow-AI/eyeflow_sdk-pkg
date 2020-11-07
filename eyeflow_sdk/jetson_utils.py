"""
    Functions to interact with NVidia Jetson Hardware
"""

import subprocess
#----------------------------------------------------------------------------------------------------------------------------------

def get_jetson_module_sn():
    """
        Gets the module serial number for Jetson
    """
    cvm_addr = "0x50"
    proc = subprocess.run(["sudo", "i2cdump", "-f", "-y", "-r", "74-86", "0", cvm_addr, "b"], stdout=subprocess.PIPE)
    if proc.returncode:
        raise Exception("Fail to get serial")

    lines = proc.stdout.decode().split('\n')
    for line in lines:
        print(line)

    serial = lines[1][-6:] + lines[2][-16:-9]

    return serial
#----------------------------------------------------------------------------------------------------------------------------------
