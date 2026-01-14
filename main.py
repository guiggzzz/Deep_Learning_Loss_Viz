import subprocess
import sys

list = False

if __name__ == "__main__":
    config_name = sys.argv[1]
    print(f"Paramètre reçu : {config_name}")

    if list:
        for config in config_name:
            subprocess.run(["python", "train.py", config])
            subprocess.run(["python", "directions.py", config])
            subprocess.run(["python", "landscape.py", config])
            subprocess.run(["python", "plot.py", config])

    else:
        subprocess.run(["python", "train.py", config_name])
        subprocess.run(["python", "directions.py", config_name])
        subprocess.run(["python", "landscape.py", config_name])
        subprocess.run(["python", "plot.py", config_name])