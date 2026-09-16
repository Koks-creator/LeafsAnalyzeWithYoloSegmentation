import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import base64
import json
import random
import string
from functools import wraps
from io import BytesIO
from time import time

import cv2
import numpy as np
import requests
from config import Config
from flask import flash, render_template

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
            f"{Config.API_HOST}:{Config.API_PORT}/health",
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
@require_api_connection
def home():
    try:
        form = forms.MainForm()
        form_validation = form.validate_on_submit()
        display_results = []

        if form_validation:
            # Preparing images
            files = []
            for image_file in form.images_field.data:
                content = image_file.read()
                filename = Path(image_file.filename).name
                files.append(
                    ("files", (filename, BytesIO(content), "image/png"))
                )

            # Config sar
            # Overal params
            model_name = form.models_list_field.data
            filter_lt_px = form.filter_lt_px_field.data

            ########### Leafs detection model ########### 
            # YOLO
            yolo_conf_thr_leaf = form.yolo_conf_thr_leaf_field.data
            yolo_iou_leaf = form.yolo_iou_leaf_field.data
            # SAHI
            use_sahi_leaf = form.use_sahi_leaf_field.data
            sahi_conf_thr_leaf = form.sahi_conf_thr_leaf_field.data
            sahi_slice_height_leaf = form.sahi_slice_height_leaf_field.data
            sahi_slice_width_leaf = form.sahi_slice_width_leaf_field.data
            sahi_overlap_height_ratio_leaf = form.sahi_overlap_height_ratio_leaf_field.data
            sahi_overlap_width_ratio_leaf = form.sahi_overlap_width_ratio_leaf_field.data
            sahi_match_thr_leaf = form.sahi_match_thr_leaf_field.data

            ########### Leaf disease detection model ###########
            # YOLO
            yolo_conf_thr_dis = form.yolo_conf_thr_dis_field.data
            yolo_iou_dis = form.yolo_iou_dis_field.data
            # SAHI
            use_sahi_dis = form.use_sahi_dis_field.data
            sahi_conf_thr_dis = form.sahi_conf_thr_dis_field.data
            sahi_slice_height_dis = form.sahi_slice_height_dis_field.data
            sahi_slice_width_dis = form.sahi_slice_width_dis_field.data
            sahi_overlap_height_ratio_dis = form.sahi_overlap_height_ratio_dis_field.data
            sahi_overlap_width_ratio_dis = form.sahi_overlap_width_ratio_dis_field.data
            sahi_match_thr_dis = form.sahi_match_thr_dis_field.data
            
            request_config = {
                "model_pair_name": model_name,
                "pad": 0.06,
                "alpha": 0.5,
                "draw_color": [200, 0, 50],
                "filter_lt_px": filter_lt_px,
                "leaf": {
                    "sahi": {
                        "use_sahi": use_sahi_leaf,
                        "conf": sahi_conf_thr_leaf,
                        "slice_height": sahi_slice_height_leaf,
                        "slice_width": sahi_slice_width_leaf,
                        "overlap_height_ratio": sahi_overlap_height_ratio_leaf,
                        "overlap_width_ratio": sahi_overlap_width_ratio_leaf,
                        "match_threshold": sahi_match_thr_leaf
                    },
                    "yolo": {
                        "conf": yolo_conf_thr_leaf,
                        "iou": yolo_iou_leaf
                    }
                },
                "disease": {
                    "sahi": {
                        "use_sahi": use_sahi_dis,
                        "conf": sahi_conf_thr_dis,
                        "slice_height": sahi_slice_height_dis,
                        "slice_width": sahi_slice_width_dis,
                        "overlap_height_ratio": sahi_overlap_height_ratio_dis,
                        "overlap_width_ratio": sahi_overlap_width_ratio_dis,
                        "match_threshold": sahi_match_thr_dis
                    },
                    "yolo": {
                        "conf": yolo_conf_thr_dis,
                        "iou": yolo_iou_dis
                    }
                }
            }

            resp = requests.post(
                f"{Config.API_HOST}:{Config.API_PORT}/process",
                files=files,
                data={"config": json.dumps(request_config)}
            )
            data_json_raw = resp.json()
            data_json = data_json_raw.get("leafs_data")
            if not data_json:
                raise FailedRequest(data_json_raw)
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
                            "leaf_id": leaf_data["leaf_id"],
                            "class_name": leaf_data["class_name"],
                            "conf": leaf_data["conf"],
                            "diseased_area_perc": leaf_data["diseased_area_perc"],
                            "n_detected": leaf_data["n_detected"]
                        }
                    )
                display_data["leafs"] = leafs_display_data
                display_results.append(
                    display_data
                )
    except Exception as e:
        app.logger.error(f"Unknown error: {e}")
        flash(f"Error: {e}", "danger")

    return render_template("home.html", form=form, display_results=display_results)
