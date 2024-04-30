import time
import os
import sys

local_path = os.path.dirname(os.path.realpath(__file__))

print("local_path: ", local_path)

status = sys.argv[1]

file_name = "object_tracking.py"

if status == "1":
    print("Starting Object Detection script")
    cmd = f"sudo python3 {local_path}/{file_name} &"
    print("cmd: ", cmd)
    os.system(cmd)
    time.sleep(1)

if status == "0":
    cmd = f"sudo pkill -f {file_name}"
    os.system(cmd)
