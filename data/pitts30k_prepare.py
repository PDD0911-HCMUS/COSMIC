"""Run Pitts30k OSG export when the current GPU job has released its memory."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time

import psutil
import torch

from config import load_config
from data.config import PROJECT_ROOT
from data.osg_storage import write_json_atomic
from data.pitts30k_export import prepare_pitts30k_osg
from data.pitts30k_metadata import read_pitts30k_metadata


def gpu_free_mib(device):
    index = device.split(":", 1)[1] if ":" in device else "0"
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
                             "--id=" + index], capture_output=True, text=True, check=True, timeout=15)
    return int(result.stdout.strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pitts30k.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--min-free-mib", type=int, default=6000)
    parser.add_argument("--check-every", type=int, default=30)
    args = parser.parse_args(argv)
    if args.check_every <= 0 or args.check_every > 60 or args.min_free_mib < 0:
        raise ValueError("check-every must be in [1,60]; min-free-mib must be nonnegative")
    if args.wait_for_pid is not None and args.wait_for_pid <= 0:
        raise ValueError("wait-for-pid must be positive")
    torch.set_num_threads(2)
    config = replace(load_config(args.config), device=args.device)
    if config.data.dataset != "pitts30k":
        raise ValueError("Pitts30k export requires data.dataset=pitts30k")
    metadata = read_pitts30k_metadata(config.data.root)
    status_path = config.output.root / "osg_job.json"
    status = {"pid":os.getpid(), "config":str(args.config.resolve()), "root":str(metadata.root),
              "device":args.device, "wait_for_pid":args.wait_for_pid,
              "started_at":datetime.now(timezone.utc).isoformat()}
    blocker = None
    if args.wait_for_pid is not None:
        try:
            blocker = psutil.Process(args.wait_for_pid)
            blocker.create_time()  # psutil.is_running also detects PID reuse.
        except psutil.NoSuchProcess:
            pass
    try:
        if args.device.startswith("cuda"):
            while True:
                free = gpu_free_mib(args.device)
                blocked = blocker is not None and blocker.is_running()
                if not blocked and free >= args.min_free_mib:
                    break
                status.update(state="waiting_for_gpu", free_mib=free, blocking_process_alive=blocked,
                              updated_at=datetime.now(timezone.utc).isoformat())
                write_json_atomic(status_path,status)
                print(f"Waiting for GPU: free={free} MiB; need={args.min_free_mib} MiB; "
                      f"current export alive={blocked}",flush=True)
                time.sleep(args.check_every)
        status.update(state="running",updated_at=datetime.now(timezone.utc).isoformat())
        write_json_atomic(status_path,status)
        result = prepare_pitts30k_osg(data_root=config.data.root,osg_root=config.data.osg_root,
                                     max_images=config.export.max_items,overwrite=config.export.overwrite,
                                     object_threshold=config.export.object_threshold,
                                     relation_threshold=config.export.relation_threshold,
                                     device=config.device,graph_format=config.export.graph_format)
        status.update(state="complete",updated_at=datetime.now(timezone.utc).isoformat(),
                      result={key:str(value) if isinstance(value,Path) else value for key,value in result.items()})
        write_json_atomic(status_path,status)
    except BaseException as error:
        status.update(state="failed",error=str(error),updated_at=datetime.now(timezone.utc).isoformat())
        write_json_atomic(status_path,status)
        raise


if __name__ == "__main__":
    main()
