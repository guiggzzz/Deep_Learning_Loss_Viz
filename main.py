import os
import subprocess
import sys

if __name__ == "__main__":
    config_name = sys.argv[1]

    if config_name =="all":
        config_names = [
           os.path.splitext(f)[0] for f in os.listdir("configs")
           if f.endswith(".yaml") 
        ]
    else:
        config_names = [config_name]
    print(f"Paramètre reçu : {config_name}")

    for cfg in config_names:
        subprocess.run(["python", "train.py", cfg])
        subprocess.run(["python", "directions.py", cfg])
        subprocess.run(["python", "landscape.py", cfg])
        subprocess.run(["python", "plot.py", cfg])