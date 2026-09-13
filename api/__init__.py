import os
import sys
from logging import Logger, getLogger
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import json

from config import Config
from custom_logger import CustomLogger
from fastapi import FastAPI
from leaf_disease_analyzer import (
    DetectionConfig,
    LeafsDiseaseAnalyzer,
    ProcessImageConfig,
    SahiConfig,
    YoloConfig,
)


# try:
def setup_logging() -> Logger:
    """Configure logging for the api"""
    log_dir = os.path.dirname(Config.API_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = CustomLogger(
        logger_name="middleware_logger",
        logger_log_level=Config.CLI_LOG_LEVEL,
        file_handler_log_level=Config.FILE_LOG_LEVEL,
        log_file_name=Config.API_LOG_FILE
    ).create_logger()

    return logger

logger = getLogger("middleware_logger")

if not logger.handlers:
    logger = setup_logging()

logger.info("Loading models")
models = {
    "model1": LeafsDiseaseAnalyzer(
        leafs_model_path=Config.YOLO_LEAFS_MODEL_FOLDER / Config.YOLO_LEAFS_MODEL_NAME,
        disease_model_path=Config.YOLO_DISEASE_MODEL_FOLDER / Config.YOLO_DISEASE_MODEL_NAME,
        device=Config.YOLO_DEVICE
    )
}

# Starting api
logger.info("Starting api")
app = FastAPI(title="LeafsDiseaseAnalyzerApi")

# sth was fucked men with files input
from fastapi.openapi.utils import get_openapi


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )

    for component in schema.get("components", {}).get("schemas", {}).values():
        for prop in component.get("properties", {}).values():

            # list[UploadFile]
            if prop.get("type") == "array":
                items = prop.get("items", {})

                if items.get("contentMediaType") == "application/octet-stream":
                    items.pop("contentMediaType", None)
                    items["format"] = "binary"

            # UploadFile
            elif (
                prop.get("type") == "string"
                and prop.get("contentMediaType") == "application/octet-stream"
            ):
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi

from api import routes
logger.info("Api initialized")
# except Exception as e:
#     logger.error(f"Error on init: {e}", exc_info=True)
#     raise e
