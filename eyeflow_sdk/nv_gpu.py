"""
    Functions to interact with NVidia GPUs
"""

import arrow
import psutil
import pynvml as nv
import subprocess
import time
import xmltodict

from contextlib import contextmanager

from pynvml import nvmlInit, nvmlShutdown

from eyeflow_sdk.log_obj import log
#----------------------------------------------------------------------------------------------------------------------------------

@contextmanager
def nvml_context():
    nvmlInit()
    yield
    nvmlShutdown()
#----------------------------------------------------------------------------------------------------------------------------------


def parse_cmd_roughly(args):
    cmdline = ' '.join(args)
    if 'python -m ipykernel_launcher' in cmdline:
        return 'jupyter'
    python_script = [arg for arg in args if arg.endswith('.py')]
    if len(python_script) > 0:
        return python_script[0]
    else:
        return cmdline if len(cmdline) <= 25 else cmdline[:25] + '...'
#----------------------------------------------------------------------------------------------------------------------------------


def device_status(device_index):
    handle = nv.nvmlDeviceGetHandleByIndex(device_index)
    device_name = nv.nvmlDeviceGetName(handle)
    device_name = device_name.decode('UTF-8')
    nv_procs = nv.nvmlDeviceGetComputeRunningProcesses(handle)
    utilization = nv.nvmlDeviceGetUtilizationRates(handle).gpu
    clock_mhz = nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)
    temperature = nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)
    pids = []
    users = []
    dates = []
    cmd = None
    for nv_proc in nv_procs:
        pid = nv_proc.pid
        pids.append(pid)
        try:
            proc = psutil.Process(pid)
            users.append(proc.username())
            dates.append(proc.create_time())
            if cmd is None:
                cmd = parse_cmd_roughly(proc.cmdline())
        except psutil.NoSuchProcess:
            users.append('?')
    return {
        'type': device_name,
        'is_available': len(pids) == 0,
        'pids': ','.join([str(pid) for pid in pids]),
        'users': ','.join(users),
        'running_since': arrow.get(min(dates)).humanize() if len(dates) > 0 else None,
        'utilization': utilization,
        'clock_mhz': clock_mhz,
        'temperature': temperature,
        'cmd': cmd,
        }
#----------------------------------------------------------------------------------------------------------------------------------


def device_statuses():
    with nvml_context():
        device_count = nv.nvmlDeviceGetCount()
        return [device_status(device_index) for device_index in range(device_count)]
#----------------------------------------------------------------------------------------------------------------------------------


def _run_cmd(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception(f"Fail executing command: {cmd} - {result.stderr}")

    return result.stdout.decode("utf-8")
#----------------------------------------------------------------------------------------------------------------------------------


def gpu_info():
    gpu_info = xmltodict.parse(_run_cmd(['nvidia-smi', '--query', '--xml-format']))["nvidia_smi_log"]["gpu"]

    if not isinstance(gpu_info, list):
        gpu_info = [gpu_info]

    for idx,_ in enumerate(gpu_info):
        mem_used = int(gpu_info[idx]["fb_memory_usage"]["used"].strip().replace('MiB', ''))
        mem_total = int(gpu_info[idx]["fb_memory_usage"]["total"].strip().replace('MiB', ''))
        gpu_info[idx]['index'] = str(idx)
        gpu_info[idx]['mem_used'] = mem_used
        gpu_info[idx]['mem_total'] = mem_total
        gpu_info[idx]['mem_used_percent'] = 100. * mem_used / mem_total

    return gpu_info
#----------------------------------------------------------------------------------------------------------------------------------


def available_gpus(free_memory=4000):
    return [gpu['index'] for gpu in gpu_info() if (gpu['mem_total'] - gpu['mem_used']) >= free_memory]
#----------------------------------------------------------------------------------------------------------------------------------


def wait_available_gpu(free_memory=4000):
    all_gpus = gpu_info()
    if not len(all_gpus):
        log.warning("None GPU available")
        raise Exception("None GPU available")

    # GLOBAL_GPU_LOCK.acquire()
    while True:
        all_gpus = gpu_info()
        gpu_list = sorted([(gpu['index'], gpu['mem_total'] - gpu['mem_used']) for gpu in all_gpus if (gpu['mem_total'] - gpu['mem_used']) >= free_memory], key=lambda x: x[1])
        if len(gpu_list) == 0:
            log.info("GPU wait 30 seconds")
            time.sleep(10)
            continue

        best_gpu = int(gpu_list[-1][0])
        with nvml_context():
            gpu_data = device_status(best_gpu)
            for gpu in all_gpus:
                if int(gpu["index"]) == best_gpu:
                    gpu_data.update(gpu)
                    break

        # GLOBAL_GPU_LOCK.release()
        log.info(f"Best GPU: {gpu_data}")
        return gpu_data
#----------------------------------------------------------------------------------------------------------------------------------
