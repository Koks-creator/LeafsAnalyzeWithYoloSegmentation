
import logging
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import Config

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

def setup_logging(app: Flask) -> None:
    """Configure logging for the application"""
    log_dir = os.path.dirname(Config.WEB_APP_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    file_handler = logging.FileHandler(Config.WEB_APP_LOG_FILE)
    file_handler.setLevel(Config.FILE_LOG_LEVEL)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - Line: %(lineno)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    
    app.logger.addHandler(file_handler)
    app.logger.setLevel(Config.FILE_LOG_LEVEL)
    app.logger.propagate = False  # Prevent duplicate logs


def create_app() -> tuple[Flask]:
    """Initialize and configure the Flask application"""
    app = Flask(
        "LeafHealthApp",
        template_folder=TEMPLATE_DIR,
        static_folder=STATIC_DIR
    )
    
    setup_logging(app)
    app.logger.info("Starting web app")
    
    app.secret_key = os.getenv("SECRET_KEY")
    app.config["TESTING"] = Config.WEB_APP_TESTING
    
    return app

app = create_app()
try:
    app.logger.info(f"Checking api connection: {Config.API_HOST}:{Config.API_PORT}/health")
    req = requests.get(f"{Config.API_HOST}:{Config.API_PORT}/health")
    if req.json()["status"] != "all green":
        app.logger.error(f"Error when checking api status: {req.json()['status']}")
    else:
        app.logger.info("Connected to api")

except Exception as e:
    app.logger.error(f"Error when connecting to the api: {e}")


os.makedirs(Config.WEB_APP_TEMP_UPLOADS_FOLDER, exist_ok=True)
from webapp import routes

app.logger.info("App started")
