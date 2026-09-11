import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import asyncio
import base64

import cv2
import numpy as np
from fastapi import Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, ValidationError

from api import (
    DetectionConfig,
    ProcessImageConfig,
    SahiConfig,
    YoloConfig,
    app,
    logger,
    models,
)

################################### Input Models ###################################

class SahiConfigModel(BaseModel):
    use_sahi: bool = False
    conf: float = 0.1
    slice_height: int = 240
    slice_width: int = 240
    overlap_height_ratio: float = 0.3
    overlap_width_ratio: float = 0.3
    match_threshold: float = 0.4


class YoloConfigModel(BaseModel):
    conf: float = 0.1
    iou: float = 0.1


class DetectionConfigModel(BaseModel):
    sahi: SahiConfigModel = Field(default_factory=SahiConfigModel)
    yolo: YoloConfigModel = Field(default_factory=YoloConfigModel)


class ProcessImageConfigModel(BaseModel):
    model_pair_name: str = "model1"
    pad: float = 0.06
    alpha: float = 0.5
    draw_color: tuple[int, int, int] = (200, 0, 50)
    filter_lt_px: int = 50000

    leaf: DetectionConfigModel = Field(default_factory=DetectionConfigModel)
    disease: DetectionConfigModel = Field(default_factory=DetectionConfigModel)

################################### Responses ###################################

class LeafResult(BaseModel):
    leaf_id: int
    class_name: str
    conf: float
    leaf_px: int
    les_px: int
    diseased_area_perc: float
    bbox: tuple[int, int, int, int]
    n_detected: int
    leaf_image: str # base64

class SingleImageResponse(BaseModel):
    draw_img: str # bytes
    leafs_data: list[LeafResult]

class Response(BaseModel):
    proc_time: float
    items_len: int
    leafs_data: list[SingleImageResponse]

class HealthResponse(BaseModel):
    status: str

################################### Helper stuff ###################################

def parse_config(config: str = Form(default="{}")) -> ProcessImageConfigModel:
    try:
        return ProcessImageConfigModel.model_validate_json(config)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

def numpy_to_base64(img_np: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_np)

    return base64.b64encode(buf).decode("utf-8")


################################### Routes ###################################

@app.get("/")
async def alive():
    return "Hello, I'm alive :) https://www.youtube.com/watch?v=9DeG5WQClUI"

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    return HealthResponse(status="all green")

@app.post("/process/", response_model=Response)
async def process(
    files: list[UploadFile] = File(...), # czemu mi to zjebie podkreslasz?
    config: ProcessImageConfigModel = Depends(parse_config)
):  
    logger.info(f"Uploaded: {len(files)} files.")
    try:
        process_config = ProcessImageConfig(
            pad=config.pad,
            alpha=config.alpha,
            draw_color=config.draw_color,
            filter_lt_px=config.filter_lt_px,
            leaf=DetectionConfig(
                sahi=SahiConfig(**config.leaf.sahi.model_dump()),
                yolo=YoloConfig(**config.leaf.yolo.model_dump()),
            ),
            disease=DetectionConfig(
                sahi=SahiConfig(**config.disease.sahi.model_dump()),
                yolo=YoloConfig(**config.disease.yolo.model_dump()),
            ),
        )
        images = []
        for file in files:
            content = await file.read()
            nparr = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            images.append(image)

        model_pair = config.model_pair_name
        model = models.get(model_pair)
        if not model:
            raise HTTPException(status_code=404, detail=f"No such model: {model_pair}.")

        logger.info(f"About to process sar with: {model_pair} model.")
        (draw_imgs, results), proc_time = await asyncio.to_thread(model.process_image, images, process_config)
        logger.info(f"Process done sar with: {model_pair} model.")

        processed_data = []
        for draw_img, leafs in zip(draw_imgs, results):
            encoded_img = numpy_to_base64(draw_img)
            jsonable_results = [
                LeafResult(
                    leaf_id=leaf.leaf_id,
                    class_name=leaf.class_name,
                    conf=leaf.conf,
                    leaf_px=leaf.leaf_px,
                    les_px=leaf.les_px,
                    diseased_area_perc=leaf.diseased_area_perc,
                    bbox=leaf.bbox,
                    n_detected=leaf.n_detected,
                    leaf_image=numpy_to_base64(leaf.img_draw)
                ) 
                for leaf in leafs
            ]
            processed_data.append(
                SingleImageResponse(
                    draw_img=encoded_img,
                    leafs_data=jsonable_results
                )
            )
        logger.info(f"Returning {len(processed_data)} results.")

        return Response(
            proc_time=proc_time,
            items_len=len(draw_imgs),
            leafs_data=processed_data
        )
    except HTTPException as http_ex:
            logger.error(f"HTTPException {http_ex}")
            raise http_ex
    except Exception as e:
        logger.error(f"(status code 500) Internal server error {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error {e}")