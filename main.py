import subprocess
import sys

list = False

if __name__ == "__main__":
    config_name = sys.argv[1]
    print(f"Paramètre reçu : {config_name}")

    if list:
        for config in config_name:
            subprocess.run(["python", "training/train.py", config])
            subprocess.run(["python", "landscape/directions.py", config])
            subprocess.run(["python", "landscape/landscape.py", config])
            subprocess.run(["python", "landscape/plot.py", config])

    else:
        subprocess.run(["python", "training/train.py", config_name])
        subprocess.run(["python", "landscape/directions.py", config_name])
        subprocess.run(["python", "landscape/landscape.py", config_name])
        subprocess.run(["python", "landscape/plot.py", config_name])