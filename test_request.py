import base64
from io import BytesIO

import cv2
import numpy as np
import requests
from config import Config

files = []
paths = [str(Config.IMAGES_FOLDER / "kantong_sakit_7.png")]
for file_path in paths:
    with open(file_path, "rb") as f:
        file_content = f.read()
    files.append(
        ("files", (file_path, BytesIO(file_content), "image/png"))
    )

config = {
    "model_pair_name": "model1",
    "pad": 0.06,
    "alpha": 0.5,
    "draw_color": [200, 0, 50],
    "filter_lt_px": 50000,
    "leaf": {
        "sahi": {
            "use_sahi": False,
            "conf": 0.1,
            "slice_height": 240,
            "slice_width": 240,
            "overlap_height_ratio": 0.3,
            "overlap_width_ratio": 0.3,
            "match_threshold": 0.4
        },
        "yolo": {
            "conf": 0.1,
            "iou": 0.1
        }
    },
    "disease": {
        "sahi": {
            "use_sahi": False,
            "conf": 0.1,
            "slice_height": 240,
            "slice_width": 240,
            "overlap_height_ratio": 0.3,
            "overlap_width_ratio": 0.3,
            "match_threshold": 0.4
        },
        "yolo": {
            "conf": 0.1,
            "iou": 0.1
        }
    }
}

resp = requests.post(
    "http://127.0.0.1:5000/process/",
    files=files,
    data={"config": {}}
)

try:
    data = resp.json()["leafs_data"]
    for d in data:
        img_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(d["draw_img"]), np.uint8), cv2.IMREAD_COLOR)

        cv2.imshow("res", img_bgr)

        for i, lol in enumerate(d["leafs_data"]):
            xddd = cv2.imdecode(np.frombuffer(base64.b64decode(lol["leaf_image"]), np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow(f"res{i}", xddd)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    print(resp.json())
