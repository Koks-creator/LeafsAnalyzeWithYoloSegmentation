import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import asyncio
import base64
from typing import Annotated

import cv2
import numpy as np
from fastapi import Depends, File, Form, HTTPException, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from api import (
    Config,
    DetectionConfig,
    ProcessImageConfig,
    SahiConfig,
    YoloConfig,
    app,
    logger,
    models,
)

################################### Input Models ###################################
Prob = Annotated[float, Field(ge=0.0, le=1.0)]
Channel = Annotated[int, Field(ge=0, le=255)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class SahiConfigModel(StrictModel):
    use_sahi: bool = False
    conf: Prob = 0.1
    slice_height: int = Field(240, ge=32, le=4096)
    slice_width: int = Field(240, ge=32, le=4096)
    overlap_height_ratio: float = Field(0.3, ge=0.0, lt=1.0)
    overlap_width_ratio: float = Field(0.3, ge=0.0, lt=1.0)
    match_threshold: Prob = 0.4

    @model_validator(mode="after")
    def check_effective_stride(self):
        if self.use_sahi:
            stride_h = self.slice_height * (1 - self.overlap_height_ratio)
            stride_w = self.slice_width * (1 - self.overlap_width_ratio)
            if min(stride_h, stride_w) < 32:
                raise ValueError(
                    "overlap ratios are too high for these slice sizes "
                    f"(effective stride {stride_h:.0f}x{stride_w:.0f} px, min 32)"
                )
        return self

class YoloConfigModel(StrictModel):
    conf: Prob = 0.1
    iou: Prob = 0.1

class DetectionConfigModel(BaseModel):
    sahi: SahiConfigModel = Field(default_factory=SahiConfigModel)
    yolo: YoloConfigModel = Field(default_factory=YoloConfigModel)

class ProcessImageConfigModel(StrictModel):
    model_pair_name: str = "model_s"
    pad: float = Field(0.06, ge=0.0, le=0.5)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    draw_color: tuple[Channel, Channel, Channel] = (200, 0, 50)
    filter_lt_px: int = Field(50_000, ge=0)

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

class FullResponse(BaseModel):
    proc_time: float
    items_len: int
    leafs_data: list[SingleImageResponse]

class HealthResponse(BaseModel):
    status: str

################################### Helper stuff/validation shit ###################################
def numpy_to_base64(img_np: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_np)

    return base64.b64encode(buf).decode("utf-8")

def _prefix_loc(errors: list[dict], prefix: tuple) -> list[dict]:
    out = []
    for err in errors:
        err = dict(err)
        err["loc"] = prefix + tuple(err.get("loc", ()))
        out.append(err)
    return out

def parse_config(config: str = Form(default="{}")) -> ProcessImageConfigModel:
    try:
        parsed = ProcessImageConfigModel.model_validate_json(config)
    except ValidationError as e:
        # include_context=False -> ctx bywa nieserializowalne (obiekty wyjątków)
        raise RequestValidationError(
            _prefix_loc(
                e.errors(include_url=False, include_context=False),
                ("body", "config"),
            )
        ) from e

    if parsed.model_pair_name not in models:
        raise RequestValidationError([{
            "type": "value_error",
            "loc": ("body", "config", "model_pair_name"),
            "msg": f"unknown model pair, available: {sorted(models)}",
            "input": parsed.model_pair_name,
        }])

    return parsed

def _file_error(idx: int, msg: str, value):
    return RequestValidationError([{
        "type": "value_error",
        "loc": ("body", "files", idx),
        "msg": msg,
        "input": value,
    }])

async def load_images(files: list[UploadFile] = File(...)) -> list[np.ndarray]:
    if not files:
        raise _file_error(0, "at least one file is required", None)
    if len(files) > Config.API_MAX_FILE_NUMBER:
        raise _file_error(0, f"too many files, max is {Config.API_MAX_FILE_NUMBER}", None)

    images = []
    for idx, file in enumerate(files):
        if file.content_type not in Config.API_ALLOWED_EXTENSIONS:
            raise _file_error(idx, f"unsupported content type, allowed: {sorted(Config.API_ALLOWED_EXTENSIONS)}", file.content_type)

        content = await file.read()
        if len(content) > Config.API_MAX_FILE_BYTES:
            raise _file_error(idx, f"File is too large, max {Config.API_MAX_FILE_BYTES // 1024 // 1024} MB", file.filename)

        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise _file_error(idx, "file is not a decodable image", file.filename)
        
        images.append(img)

    return images

################################### Handlers ###################################
@app.exception_handler(RequestValidationError)
async def on_validation_error(request, exc: RequestValidationError):
    errors = [
        {
            "field": ".".join(str(p) for p in err["loc"] if p != "body") or "body",
            "message": err["msg"],
            "type": err["type"],
            "input": err.get("input"),
        }
        for err in exc.errors()
    ]
    logger.warning(f"422 on {request.url.path}: {errors}")
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({
            "detail": "Request validation failed",
            "errors": errors,
        }),
    )

################################### Routes ###################################
@app.get("/")
async def alive():
    return "Hello, I'm alive :) https://www.youtube.com/watch?v=9DeG5WQClUI"

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    return HealthResponse(status="all green")

@app.post("/process/", response_model=FullResponse)
async def process(
    images: list[np.ndarray] = Depends(load_images), # czemu mi to zjebie podkreslasz?
    config: ProcessImageConfigModel = Depends(parse_config)
):  
    logger.info(f"Uploaded: {len(images)} files.")
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

        model_pair = config.model_pair_name
        model = models[model_pair]

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

        return FullResponse(
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
