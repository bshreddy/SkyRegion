import os
import requests
from tqdm import tqdm

models_dir = "models"
files = {"sky-region_mask_r-cnn_resnet50-fpn-1579167716": "https://firebasestorage.googleapis.com/v0/b/sky-regions-sih.appspot.com/o/models%2Fsky-region_mask_r-cnn_resnet50-fpn-1579167716?alt=media",
         "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
         "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"}

if not os.path.exists(models_dir):
    os.mkdir(models_dir)

for filename, fileurl in files.items():
    req = requests.get(fileurl, stream=True)
    file_size = int(req.headers["content-length"])
    chunk_size = 1000

    with open(os.path.join(models_dir, filename), "wb") as f:
        with tqdm(desc=f"Fetching {filename}", total=file_size, unit_scale=True) as pbar:
            for chunk in req.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)