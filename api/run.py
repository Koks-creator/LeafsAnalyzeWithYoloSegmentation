import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import uvicorn
from config import Config

from api import app


def run_api() -> None:
    uvicorn.run(app, 
                host=Config.API_HOST, 
                port=Config.API_PORT,
                # log_config=Config().get_uvicorn_logger()
            )

if __name__ == "__main__":
    run_api()