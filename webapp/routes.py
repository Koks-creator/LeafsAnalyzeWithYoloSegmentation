import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import base64
import random
import string
from functools import wraps
from io import BytesIO
from time import time

import cv2
import numpy as np
import requests
from config import Config
from flask import flash, render_template, request
from PIL import Image

from webapp import app, forms

_api_status_cache = {"connected": False, "last_check": 0}
API_CHECK_INTERVAL = Config.WEB_API_CHECK_INTERVAL

# Exceptions
class FailedRequest(Exception):
    pass

####

def random_string(n: int = 10) -> str:
    pool = string.ascii_lowercase + "".join(str(i) for i in range(10))
    return "".join([random.choice(pool) for _ in range(n)])


def check_api_connection() -> bool:
    """checks connection to api with some cache"""
    now = time()
    
    if now - _api_status_cache["last_check"] < API_CHECK_INTERVAL:
        return _api_status_cache["connected"]
    
    try:
        response = requests.get(
            f"http://{Config.API_HOST}:{Config.API_PORT}/health",
            timeout=5
        )
        connected = response.status_code == 200 and response.json().get("status") == "all green"
        
    except Exception as e:
        app.logger.error(f"API connection check failed: {e}")
        connected = False
    
    _api_status_cache["connected"] = connected
    _api_status_cache["last_check"] = now
    return connected

def require_api_connection(f) -> None:
    """Dekorator sprawdzający połączenie z API."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not check_api_connection():
            flash("API is not available. Try again later.", "danger")
            return render_template("home.html", form=forms.MainForm(), results=[])
        return f(*args, **kwargs)
    return decorated_function


@app.route("/health", methods=["GET"])
def health_check():
    api_status = check_api_connection()
    return {"status": api_status}


@app.route("/", methods=["GET", "POST"])
# @require_api_connection
def home():
    form = forms.MainForm()
    form_validation = form.validate_on_submit()
    display_results = []
    if form_validation:
        files = []

        for image_file in form.images_field.data:
            content = image_file.read()
            filename = Path(image_file.filename).name
            files.append(
                ("files", (filename, BytesIO(content), "image/png"))
            )

        model_name = form.models_list_field.data
        print(form.yolo_conf_thr_dis_field)
        # bedzie trza w slowniki to popakowac, wgl azdy wynik niech ma swoj folder z glownym obrazem i liscmi
        resp = requests.post("http://127.0.0.1:5000/process", files=files,  json={
                "config": {
                    "model_pair_name": "model",
                    "pad": 0.06,
                    "alpha": 0.5,
                    "draw_color": [200, 0, 50]
                }})
        data_json = resp.json()["leafs_data"]
        for ind, data in enumerate(data_json):
            display_data = {}

            timestamp = int(time())
            dir_name = f"{timestamp}_{ind}"
            dir_path = Config.WEB_APP_TEMP_UPLOADS_FOLDER / dir_name
            os.makedirs(dir_path, exist_ok=True)

            main_draw_img = cv2.imdecode(np.frombuffer(base64.b64decode(data["draw_img"]), np.uint8), cv2.IMREAD_COLOR)
            main_draw_filename = f"main_img_draw_{ind}.jpg"
            cv2.imwrite(dir_path / main_draw_filename, main_draw_img)
            display_data["main_img_draw"] = f"{dir_name}/{main_draw_filename}"

            leafs_display_data = []
            for leaf_data in data["leafs_data"]:
                leaf_draw_img = cv2.imdecode(np.frombuffer(base64.b64decode(leaf_data["leaf_image"]), np.uint8), cv2.IMREAD_COLOR)
                leaf_draw_filename = f"leaf_{leaf_data["leaf_id"]}.jpg"

                cv2.imwrite(dir_path / leaf_draw_filename, leaf_draw_img)

                leafs_display_data.append(
                    {
                        "leaf_img": f"{dir_name}/{leaf_draw_filename}",
                        "leaf_id": leaf_data["leaf_id"]
                    }
                )
            display_data["leafs"] = leafs_display_data
            display_results.append(
                display_data
            )
        # madrze jako to wyciagnac
    # print("xd")
    return render_template("home.html", form=form, display_results=display_results)