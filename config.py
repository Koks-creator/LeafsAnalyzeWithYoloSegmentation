import json
import logging
from pathlib import Path


class Config:
    # Overall
    ROOT_PATH: str = Path(__file__).resolve().parent # maybe set static path sar

    # Folders
    VIDEOS_FOLDER: Path =  ROOT_PATH / "videos"
    IMAGES_FOLDER: Path =  ROOT_PATH / "images"

    # SAHI MODEL PARAMS
    USE_SAHI: bool = False
    SAHI_CONF_THRESH: float = .2
    SAHI_SLICE_HEIGHT: int = 480
    SAHI_SLICE_WIDTH: int = 480
    SAHI_OVERLAP_HEIGHT_RATIO: float = 0.2
    SAHI_OVERLAP_WIDTH_RATIO: float = 0.2

    # YOLO Model
    YOLO_MODELS_FOLDER_PATH: Path = ROOT_PATH / "models"
    YOLO_LEAFS_MODEL_FOLDER: Path = YOLO_MODELS_FOLDER_PATH / "leafs_model_agrobotanix_s_1"
    YOLO_LEAFS_MODEL_NAME: str = "best.pt"
    YOLO_DISEASE_MODEL_FOLDER: Path = YOLO_MODELS_FOLDER_PATH / "disease_model_s_1"
    YOLO_DISEASE_MODEL_NAME: str = "best.pt"
    YOLO_DEVICE: str = "cpu"
    YOLO_IOU: float = .2
    YOLO_CONF_THRESH: float = .2
    YOLO_AUGMENT: bool = True
    YOLO_AGNOSTIC_NMS: bool = True

    # LOGGER
    UVICORN_LOG_CONFIG_PATH: Path = Path(ROOT_PATH) / "api" / "uvicorn_log_config.json"
    CLI_LOG_LEVEL: int = logging.INFO
    FILE_LOG_LEVEL: int = logging.INFO
    LOGS_PATH: Path = ROOT_PATH / "logs" / "logs.log"

    # API
    API_PORT: int = 5000
    API_HOST: str = "http://127.0.0.1"
    API_HOST_NO_PROT: str = "127.0.0.1"
    MAX_IMAGE_FILES: int = 5
    API_LOG_FILE: str = Path(ROOT_PATH) / "logs" / "api_logs.log"
    # API_MODELS_LIST_PATH: Union[str, os.PathLike, Path] = Path(ROOT_PATH) / "api" / "model_to_load.json"
    API_MODEL_MIN_GEN_LEN: int = 1
    API_MODEL_MAX_GEN_LEN: int = 32
    API_MAX_FILE_NUMBER: int = 5
    API_MAX_FILE_BYTES: int = 25 * 1024 * 1024
    API_ALLOWED_EXTENSIONS: tuple[str] = ("image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp")


    # WEB APP
    WEB_APP_PORT: int = 8000
    WEB_APP_HOST: str = "127.0.0.1"
    WEB_APP_DEBUG: bool = True
    WEB_APP_LOG_FILE: str = Path(ROOT_PATH) / "logs" / "web_app.log"
    WEB_APP_TEMP_UPLOADS_FOLDER =  Path(ROOT_PATH) / "webapp" / "static" / "temp_uploads"
    WEB_APP_FILES_LIFE_TIME: int = 60
    WEB_APP_USE_SSL: bool = False
    WEB_APP_SSL_FOLDER: str = f"{ROOT_PATH}/ocr_webapp/ssl_cert"
    WEB_APP_TESTING: bool = False
    WEB_APP_LOG_LEVEL: int = logging.DEBUG
    WEB_API_CHECK_INTERVAL: int = 10

    def get_uvicorn_logger(self) -> dict:
        with open(self.UVICORN_LOG_CONFIG_PATH) as f:
            log_config = json.load(f)
            log_config["handlers"]["file_handler"]["filename"] =  Path(self.ROOT_PATH) / "logs" / "api_logs.log"
            return log_config
