from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from random import randint
from time import perf_counter

import cv2
import numpy as np
from config import Config
from custom_decorators import log_call, timeit
from custom_logger import CustomLogger
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction, PredictionResult
from sahi.utils.cv import visualize_object_predictions
from ultralytics import YOLO
from ultralytics.engine.results import Results

logger = CustomLogger(
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL,
    log_file_name=Config.LOGS_PATH
).create_logger()

@dataclass
class YoloObjectData:
    bbox: tuple[int, int, int, int]
    class_id: int
    class_name: str
    conf: float
    mask: None | np.ndarray
    polygon: None | np.ndarray


@dataclass
class YoloSegmentation:
    model_path: Path
    device: str = "cpu"

    @timeit(logger=logger)
    def __post_init__(self) -> None:
        logger.info("Initializing YoloSegmentation with: \n" \
                       f" - {self.model_path}\n" \
                       f" - {self.device}" \
               )
        self.model = YOLO(self.model_path)
        self.colors = {}
        self.sahi_model = None

        if self.model.task != "segment":
            logger.error(f"Expected a segmentation model, got task={self.model.task} ({self.model_path})")
            raise ValueError(f"Expected a segmentation model, got task={self.model.task} ({self.model_path})")
        logger.info(f"Model loaded, {self.model.task=}")
            
    @staticmethod
    def _mask_from_pred(p: ObjectPrediction, H: int, W: int) -> np.ndarray:
        m = getattr(p, "mask", None)
        if m is None:
            return None

        seg = getattr(m, "segmentation", None)
        if seg:
            out = np.zeros((H, W), np.uint8)
            for poly in seg:
                if len(poly) >= 6:
                    pts = np.asarray(poly, np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(out, [pts], 1)
            return out.astype(bool)

        bm = getattr(m, "bool_mask", None)
        if bm is None:
            return None
        bm = np.asarray(bm, bool)
        if bm.shape == (H, W):
            return bm
        x1, y1, x2, y2 = [int(v) for v in p.bbox.to_xyxy()] 
        out = np.zeros((H, W), bool)
        out[y1:y2, x1:x2] = cv2.resize(bm.astype(np.uint8), (x2 - x1, y2 - y1),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
        return out

    def draw_bbox(self,
                frame: np.ndarray,
                x1: int,
                y1: int,
                x2: int,
                y2: int,
                class_name: str,
                color: tuple[int, int, int]
            ) -> None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, 1)
        cv2.putText(frame, f"{class_name}", (x1, y1-5), cv2.FONT_HERSHEY_PLAIN, 1.4, color, 2)

    @log_call(logger=logger, log_params=["conf", "iou", "plot"], hide_res=True)
    @timeit(logger=logger)
    def segment_with_yolo(self,
                        images: list[np.ndarray] | list[Path],
                        conf: float = .1,
                        iou: float = .1,
                        plot: bool = False
                    ) -> tuple[list, list]:
        results = []
        vis = []
        res = self.model.predict(
                source=images,
                device=self.device,
                conf=conf,
                iou=iou,
                verbose=False,
                retina_masks=True
            )
        for r in res:
            results.append(r)
            # don't use plot if you don't need it - it's kinda expensive
            vis.append(r.plot() if plot else None)
        
        return results, vis

    @log_call(logger=logger, log_params=["conf", "slice_height", "slice_width",
                                         "overlap_height_ratio", "overlap_width_ratio",
                                         "match_threshold", "plot"], hide_res=True)
    @timeit(logger=logger)
    def segment_with_sahi(self,
                        images: list[np.ndarray],
                        conf: float = 0.2,
                        slice_height: int = 240,
                        slice_width: int = 240,
                        overlap_height_ratio: float = 0.3,
                        overlap_width_ratio: float = 0.3,
                        match_threshold: float = 0.4,
                        plot: bool = False
                    ) -> list[tuple]:
        """
        slice_height, slice_width - higher = better for smaller objects
        """
        if self.sahi_model is None:
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(self.model_path), # sahi is retarded and it can't take Path like shit, gratuluję kurwa mózgu
                confidence_threshold=conf,
                device=self.device,
            )

        results = []
        vis_list = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            sahi_result = get_sliced_prediction(
                image=Image.fromarray(rgb),
                detection_model=self.sahi_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                postprocess_type="GREEDYNMM",          # NMS NIE scala masek, tylko wybiera jedna
                postprocess_match_metric="IOS",
                postprocess_match_threshold=match_threshold,
                verbose=False
            )
            results.append(sahi_result)
            if plot:
                vis = visualize_object_predictions(
                    image=rgb, object_prediction_list=sahi_result.object_prediction_list,
                    rect_th=2, text_size=0.5, text_th=2)
                vis = cv2.cvtColor(np.array(vis["image"]), cv2.COLOR_RGB2BGR)

                vis_list.append(vis)
            else:
                vis_list.append(None)
        
        return results, vis_list

    @log_call(logger=logger, log_params=[""], hide_res=True)
    @timeit(logger=logger)
    def normalize_results(self, results_raw: Results) -> Generator[YoloObjectData]:
        r = results_raw

        if r.boxes is None or len(r.boxes) == 0:
            return

        class_names = r.names
        boxes = r.boxes.xyxy.int().tolist()
        classes = r.boxes.cls.int().tolist()
        confs = r.boxes.conf
        if r.masks is not None:
            masks = r.masks.data.bool().cpu().numpy()
            masks_xy = r.masks.xy
        else:
            masks = None
            masks_xy = None

        for i in range(len(boxes)):
            box = boxes[i]
            conf = round(float(confs[i]), 2)
            class_id = classes[i]
            class_name = class_names[class_id]

            if r.masks is not None:
                mask = masks[i]
                polygon = masks_xy[i].astype("int32")
            else:
                mask = None
                polygon = None

            yield YoloObjectData(
                class_id=class_id,
                class_name=class_name,
                bbox=box,
                conf=conf,
                mask=mask,
                polygon=polygon
            )

    @log_call(logger=logger, log_params=[""], hide_res=True)
    @timeit(logger=logger)
    def normalize_sahi_results(self,
                               sahi_result_raw: PredictionResult,
                               img_hw: tuple[int, int]
                               ) -> Generator[YoloObjectData]:
        h, w = img_hw

        for p in sahi_result_raw.object_prediction_list:
            m = self._mask_from_pred(p, h, w)
                
            if m is not None:
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygon = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype("int32")
            else:
                polygon = None

            yield YoloObjectData(
                class_id=p.category.id,
                class_name=p.category.name,
                bbox=[int(v) for v in p.bbox.to_xyxy()],
                conf=round(float(p.score.value), 2),
                mask=m,
                polygon=polygon
            )

    @log_call(logger=logger, log_params=["conf", "iou", "px_per_mm", "resize",
                                         "show_img", "plot", "use_sahi", "sahi_conf",
                                         "sahi_slice_height", "sahi_slice_width",
                                         "sahi_overlap_height_ratio", "sahi_overlap_width_ratio",
                                         "sahi_match_threshold"], hide_res=True)
    @timeit(logger=logger)
    def process_image(self,
                    images: list[np.ndarray],
                    conf: float = .1,
                    iou: float = .1,
                    px_per_mm: int = 12,
                    resize: bool | tuple = (1280, 720),
                    show_img: bool = True,
                    plot: bool = False,
                    use_sahi: bool = False,
                    sahi_conf: float = 0.2,
                    sahi_slice_height: int = 240,
                    sahi_slice_width: int = 240,
                    sahi_overlap_height_ratio: float = 0.3,
                    sahi_overlap_width_ratio: float = 0.3,
                    sahi_match_threshold: float = 0.4,
                    ) -> list:

        if use_sahi:
            results, vis = self.segment_with_sahi(
                images=images,
                conf=sahi_conf,
                slice_height=sahi_slice_height,
                slice_width=sahi_slice_width,
                overlap_height_ratio=sahi_overlap_height_ratio,
                overlap_width_ratio=sahi_overlap_width_ratio,
                match_threshold=sahi_match_threshold,
                plot=plot
            )
        else:
            results, vis = self.segment_with_yolo(
                images=images,
                conf=conf,
                iou=iou,
                plot=plot
            )

        res_data = []
        for image_id, (img, res) in enumerate(zip(images, results)):
            h, w = img.shape[:2]
            image_data = []
            if use_sahi:
                normalized_data = self.normalize_sahi_results(res, img_hw=(h, w))
            else:
                normalized_data = self.normalize_results(res)
            for det in normalized_data:
                c = det.conf
                class_id = det.class_id
                mask = det.mask
                polygon = det.polygon
                class_name = det.class_name
                x1, y1, x2, y2 = det.bbox
                class_color = self.colors.get(class_name)
                if not class_color:
                    self.colors[class_name] = [randint(50, 255) for _ in range(3)]
                    class_color = self.colors[class_name]

                if mask is None:
                    continue
                mask_np = mask.astype(bool)
                area_px = int(mask.sum().item())

                area_mm2 = (
                    area_px / px_per_mm**2
                    if px_per_mm
                    else None
                )
                if area_mm2 is not None:
                    area_mm2 = round(area_mm2, 2)
                # print(area_mm2)
                area_pct = 100.0 * area_px / (h * w)

                alpha = 0.5

                img[mask_np] = (
                    img[mask_np].astype(np.float32) * (1 - alpha)
                    + np.array(class_color, np.float32) * alpha
                ).astype(np.uint8)

                self.draw_bbox(
                    frame=img,
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    class_name=f"{class_name} {c}",
                    color=class_color
                )

                image_data.append({
                        "area_pct": round(area_pct, 3),
                        "area_mm2": (
                            area_mm2
                            if area_mm2 is not None
                            else None
                        ),
                        "bbox": (x1, y1, x2, y2),
                        "conf": c,
                        "class_name": class_name
                })

            res_data.append({
                "image_id": image_id,
                "objects": image_data
            })

            if show_img:
                disp = img
                if resize:
                    disp = cv2.resize(img, resize if isinstance(resize, tuple) else (1280, 720))
                cv2.imshow("res", disp)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return res_data

        
if __name__ == "__main__":
    from config import Config
    ys = YoloSegmentation(
        model_path=r"C:\Users\table\PycharmProjects\MojeCos2\segmentyuumozebedo\models\leafs_model_agrobotanix_m_1\best.pt"
    )

    img = cv2.imread(f"{Config.IMAGES_FOLDER}/kantong_sakit_5.png")
    res, vis = ys.segment_with_yolo(
        images=[img],
        plot=True
    )

    cv2.imshow("res", vis[0])
    cv2.waitKey(0)
    # # print(len(res))
    # x = ys.process_image(images=[img])
    # print(x)

    # cap = cv2.VideoCapture("videos/12414493_1280_720_30fps.mp4")

    # start = perf_counter()
    # p_time = start
    # print(cap.isOpened())
    # while cap.isOpened():
    #     print("xd")
    #     success, frame = cap.read()
    #     if not success:
    #         print("spierdalaj")
    #         break

    #     x = ys.process_image(images=[frame], show_img=False, use_sahi=False)
    #     print(x)
    #     c_time = perf_counter()
    #     fps = int(1 / (c_time - p_time))
    #     p_time = c_time

    #     cv2.putText(frame, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (100, 0, 255), 2)
        
    #     cv2.imshow("res", frame)
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

