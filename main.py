import subprocess

list = False

config = "skip_conn"  # Replace with the desired parameter value
config_list = ["dont_skip_conn_depth6", "skip_conn_depth6","skip_conn", "dont_skip_conn"]

if list:
    for config in config_list:
        subprocess.run(["python", "training/train.py", config])
        subprocess.run(["python", "landscape/directions.py", config])
        subprocess.run(["python", "landscape/landscape.py", config])
        subprocess.run(["python", "landscape/plot.py", config])

else:
    subprocess.run(["python", "training/train.py", config])
    subprocess.run(["python", "landscape/directions.py", config])
    subprocess.run(["python", "landscape/landscape.py", config])
    subprocess.run(["python", "landscape/plot.py", config])