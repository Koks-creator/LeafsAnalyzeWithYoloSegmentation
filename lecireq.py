import requests
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import cv2

files = []
test_files = [r"C:\Users\table\PycharmProjects\MojeCos2\segmentyuumozebedo\images\kantong_sakit_7.png"]
for file_path in test_files:
    with open(file_path, "rb") as f:
        file_content = f.read()
    files.append(
        ("files", (file_path, BytesIO(file_content), "image/png"))
    )





resp = requests.post("http://127.0.0.1:5000/process", files=files, json={
  "config": {
    "model_pair_name": "model",
    "pad": 0.06,
    "alpha": 0.5,
    "draw_color": [200, 0, 50],
    "filter_lt_px": 50000,
    "leaf": {
      "sahi": {
        "use_sahi": True,
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
        "use_sahi": True,
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
})
data = resp.json()["leafs_data"]
for xd in data:
    img_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(xd["draw_img"]), np.uint8), cv2.IMREAD_COLOR)


    cv2.imshow("res", img_bgr)

    for i, lol in enumerate(xd["leafs_data"]):
        xddd = cv2.imdecode(np.frombuffer(base64.b64decode(lol["leaf_image"]), np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow(f"res{i}", xddd)
    cv2.waitKey(0)
cv2.destroyAllWindows()