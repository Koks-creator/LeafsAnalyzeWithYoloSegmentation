import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import shutil
from glob import glob
from time import time

from config import Config
from custom_decorators import timeit
from custom_logger import CustomLogger


logger = CustomLogger(
    logger_name="DeleteTempFiles",
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL,
    log_file_name=Config.ROOT_PATH / "logs" / "delete_temp_files.log"
).create_logger()


@timeit(logger=logger)
def sztarte() -> None:
    all_folders = glob(
        f"{Config.WEB_APP_TEMP_UPLOADS_FOLDER}/*"
    )
    files_life_time = Config.WEB_APP_FILES_LIFE_TIME

    deleted = 0
    failed = 0

    print(Config.WEB_APP_TEMP_UPLOADS_FOLDER)

    for folder_path in all_folders:
        try:
            if not os.path.isdir(folder_path):
                continue

            parent_path, folder_name = os.path.split(folder_path)
            folder_timestamp = folder_name.split("_")[0]

            if time() - int(folder_timestamp) > files_life_time:
                shutil.rmtree(folder_path)
                deleted += 1

        except Exception as e:
            logger.error(f"Error deleting {folder_path}, {e}")
            failed += 1

    logger.info(f"Deleted: {deleted}, failed: {failed}")


if __name__ == "__main__":
    sztarte()  # me mum died in the holly