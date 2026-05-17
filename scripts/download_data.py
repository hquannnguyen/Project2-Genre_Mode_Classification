# download_data.py
import os
import urllib.request
import tarfile
from config import config

def download_gtzan():
    os.makedirs(config.DATA_RAW, exist_ok=True)
    url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
    target = os.path.join(config.DATA_RAW, "genres.tar.gz")
    if not os.path.exists(target):
        print("Downloading GTZAN dataset...")
        urllib.request.urlretrieve(url, target)
        print("Download complete.")
    else:
        print("File already exists.")

def extract_gtzan():
    tar_path = os.path.join(config.DATA_RAW, "genres.tar.gz")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(config.DATA_GENRES)
    print(f"Extracted to {config.DATA_GENRES}")

if __name__ == "__main__":
    download_gtzan()
    extract_gtzan()