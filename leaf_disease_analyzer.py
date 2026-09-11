from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from config import Config
from custom_decorators import timeit
from custom_logger import CustomLogger
from yolo_segm import YoloSegmentation

logger = CustomLogger(
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL,
    log_file_name=Config.LOGS_PATH
).create_logger()


class ModelInitError(Exception):
    pass

@dataclass
class SahiConfig:
    use_sahi: bool = False
    conf: float = .1
    slice_height: int = 240
    slice_width: int = 240
    overlap_height_ratio: float = 0.3
    overlap_width_ratio: float = 0.3
    match_threshold: float = 0.4

@dataclass
class YoloConfig:
    conf: float = .1
    iou: float = .1

@dataclass
class DetectionConfig:
    sahi: SahiConfig = field(default_factory=SahiConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)

@dataclass
class ProcessImageConfig:
    pad: float = 0.06
    alpha: float = 0.5
    draw_color: tuple[int, int, int] = (200, 0, 50)
    filter_lt_px: int = 50000

    leaf: DetectionConfig = field(default_factory=DetectionConfig)
    disease: DetectionConfig = field(default_factory=DetectionConfig)

@dataclass
class LeafResult:
    leaf_id: int
    img_org: np.ndarray
    img_draw: np.ndarray
    class_name: str
    conf: float
    leaf_px: int
    les_px: int
    diseased_area_perc: float
    bbox: tuple[int, int, int, int]
    n_detected: int


@dataclass
class LeafsDiseaseAnalyzer:
    leafs_model_path: Path = Config.YOLO_LEAFS_MODEL_FOLDER / Config.YOLO_LEAFS_MODEL_NAME
    disease_model_path: Path = Config.YOLO_DISEASE_MODEL_FOLDER / Config.YOLO_DISEASE_MODEL_NAME
    device: str = Config.YOLO_DEVICE

    @timeit(logger=logger)
    def __post_init__(self) -> None:
        logger.info("Initializing LeafsDiseaseAnalyzer with: \n" \
                f" - {self.leafs_model_path}\n" \
                f" - {self.disease_model_path}\n" \
                f" - {self.device}" \
        )

        try:
            self.leafs_model_det = YoloSegmentation(
                model_path=self.leafs_model_path,
                device=self.device
            )

            self.disease_model_det = YoloSegmentation(
                model_path=self.disease_model_path,
                device=self.device
            )
        except Exception as e:
            logger.error(f"Error when initing models: {e}", exc_info=True)
            raise ModelInitError("Failed to init models")

    def draw_bbox(self,
                frame: np.ndarray,
                x1: int,
                y1: int,
                x2: int,
                y2: int,
                class_name: str,
                color: tuple[int, int, int],
                rect_th: int = 2,
                font_th: int = 2,
                font_scale: int = 1.4,
                pad: int = 5
            ) -> None:
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, rect_th, 1)
        cv2.putText(frame, f"{class_name}", (x1, y1-pad), cv2.FONT_HERSHEY_PLAIN, font_scale, color, font_th)

    @staticmethod
    def cv2_style(img: np.ndarray) -> dict:
        scale = min(img.shape[:2]) / 1080

        return {
            "font_scale": 3 * scale,
            "text_thickness": max(1, round(4 * scale)),
            "bbox_thickness": max(1, round(6 * scale)),
            "padding": max(1, round(5 * scale)),
        }

    @staticmethod
    def _chunks(seq: any, n: int):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    @timeit(logger=logger, return_val=True)
    def process_image(self,
            images: np.ndarray,
            config: ProcessImageConfig | None = None
            ) -> tuple[np.ndarray, list[LeafResult]]:
        draw_images = []
        final_results = []

        config = config or ProcessImageConfig()

        pad = config.pad
        alpha = config.alpha
        draw_color = config.draw_color
        filter_lt_px = config.filter_lt_px
        leaf_cfg_sahi = config.leaf.sahi
        leaf_cfg_yolo = config.leaf.yolo
        disease_cfg_sahi = config.disease.sahi
        disease_cfg_yolo = config.disease.yolo

        if not leaf_cfg_sahi.use_sahi:
            leaf_res, _ = self.leafs_model_det.segment_with_yolo(
                images=images,
                plot=False,
                conf=leaf_cfg_yolo.conf,
                iou=leaf_cfg_yolo.iou
            )
        else:
            leaf_res, _ = self.leafs_model_det.segment_with_sahi(
                images=images,
                conf=leaf_cfg_sahi.conf,
                slice_height=leaf_cfg_sahi.slice_height,
                slice_width=leaf_cfg_sahi.slice_width,
                overlap_height_ratio=leaf_cfg_sahi.overlap_height_ratio,
                overlap_width_ratio=leaf_cfg_sahi.overlap_width_ratio,
                match_threshold=leaf_cfg_sahi.match_threshold,
                plot=False
            )

        for img, res in zip(images, leaf_res):
            draw_img = img.copy()
            h, w, _ = img.shape

            if not leaf_cfg_sahi.use_sahi:
                normalized_data = self.leafs_model_det.normalize_results(res)
            else:
                normalized_data = self.leafs_model_det.normalize_sahi_results(res, img_hw=(h, w))

            leafs_data = []

            whole_img_styles = self.cv2_style(img=img)

            for i, data in enumerate(normalized_data):
                leaf = data.mask
                if leaf is None or leaf.sum() > filter_lt_px:
                    continue
                x1, y1, x2, y2 = data.bbox
                # print(x1, y1, x2, y2)
                dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
                x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
                x2, y2 = min(w, int(x2 + dx)), min(h, int(y2 + dy))
                class_name = data.class_name
                conf = data.conf

                # 102911 px - to duze gowno
                """
                20271
                23707
                14547
                28129

                listki tyle maja px
                """

                leaf_c = leaf[y1:y2, x1:x2]
                crop = (img[y1:y2, x1:x2] * leaf_c[:, :, None]).astype(np.uint8)

                self.draw_bbox(
                    frame=draw_img,
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    class_name=f"{class_name} {conf}",
                    color=draw_color,
                    rect_th=whole_img_styles["bbox_thickness"],
                    font_scale=whole_img_styles["font_scale"],
                    font_th=whole_img_styles["text_thickness"],
                    pad=whole_img_styles["padding"]
                )

                # bbox tez itp
                draw_img[leaf] = (
                    draw_img[leaf].astype(np.float32) * (1 - alpha)
                    + np.array(draw_color, np.float32) * alpha
                ).astype(np.uint8)

                leafs_data.append({
                    "img": crop,
                    "leaf_c": leaf_c,
                    "class_name": class_name,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2)
                })

            results = []
                
            if not leafs_data:
                final_results.append([])
                draw_images.append(draw_img)
                continue
            disease_norm = []

            # we doing some chunking and shieeet cuz in fast api it was foking up with RAM daaaaamn, went from 2GB to 600-700mb DAAAAMN SON
            for chunk in self._chunks(seq=leafs_data, n=4):
                if not disease_cfg_sahi.use_sahi:
                    preds, _ = self.disease_model_det.segment_with_yolo(
                        images=[r["img"] for r in chunk], plot=False,
                        conf=disease_cfg_yolo.conf, iou=disease_cfg_yolo.iou
                    )
                    disease_norm.extend(
                        [self.disease_model_det.normalize_results(dis_pred) for dis_pred in preds]
                    )
                else:
                    preds, _ = self.disease_model_det.segment_with_sahi(
                        images=[r["img"] for r in chunk],
                        plot=False,
                        conf=disease_cfg_sahi.conf,
                        slice_height=disease_cfg_sahi.slice_height,
                        slice_width=disease_cfg_sahi.slice_width,
                        overlap_height_ratio=disease_cfg_sahi.overlap_height_ratio,
                        overlap_width_ratio=disease_cfg_sahi.overlap_width_ratio,
                        match_threshold=disease_cfg_sahi.match_threshold
                    )

                    disease_norm.extend(
                        self.disease_model_det.normalize_sahi_results(p, img_hw=r["img"].shape[:2])
                        for p, r in zip(preds, chunk)
                    )
                del preds
            # tu wyzej

            for i, (normalized_data_dis, leaf_data) in enumerate(zip(disease_norm, leafs_data)):
                cropped_leaf = leaf_data["img"]
                cropped_leaf_draw = cropped_leaf.copy()
                class_name = leaf_data["class_name"]
                bbox = leaf_data["bbox"]
                conf = leaf_data["conf"]
                leaf_c = leaf_data["leaf_c"]
                les_c = np.zeros(leaf_c.shape, bool)
                n = 0

                for data in normalized_data_dis:
                    mask_raw = data.mask
                    if mask_raw is None:
                        continue
                    m = mask_raw.astype(bool)
                    if m.shape != leaf_c.shape:
                        continue
                    les_c |= m
                    n += 1

                    x1_d, y1_d, x2_d, y2_d = data.bbox
                    class_name_d = data.class_name
                    conf_d = data.conf

                    leaf_img_styles = self.cv2_style(img=cropped_leaf)
                    self.draw_bbox(
                        frame=cropped_leaf_draw,
                        x1=x1_d,
                        x2=x2_d,
                        y1=y1_d,
                        y2=y2_d,
                        class_name=f"{class_name_d} {conf_d}",
                        color=draw_color,
                        rect_th=leaf_img_styles["bbox_thickness"],
                        font_scale=leaf_img_styles["font_scale"],
                        font_th=leaf_img_styles["text_thickness"],
                        pad=whole_img_styles["padding"]
                    )

                    # bbox tez itp
                    cropped_leaf_draw[mask_raw] = (
                        cropped_leaf_draw[mask_raw].astype(np.float32) * (1 - alpha)
                        + np.array(draw_color, np.float32) * alpha
                    ).astype(np.uint8)

                les_c &= leaf_c
                leaf_px, les_px = int(leaf_c.sum()), int(les_c.sum())
                diseased_area_perc = round(100 * les_px / leaf_px, 2) if leaf_px else 0.0

                results.append(
                    LeafResult(
                        leaf_id=i,
                        img_org=cropped_leaf,
                        img_draw=cropped_leaf_draw,
                        class_name=class_name,
                        conf=conf,
                        leaf_px=leaf_px,
                        les_px=les_px,
                        diseased_area_perc=diseased_area_perc,
                        bbox=bbox,
                        n_detected=n
                    )
                )
            final_results.append(results)
            draw_images.append(draw_img)

        return draw_images, final_results


if __name__ == "__main__":
    config = ProcessImageConfig(
        pad=0.06,
        filter_lt_px=50000,
        leaf=DetectionConfig(
            sahi=SahiConfig(
                use_sahi=False,
                conf=0.15,
                slice_height=240
            ),
            yolo=YoloConfig(
                conf=0.2,
                iou=0.3
                )
            ),
            disease=DetectionConfig(
                sahi=SahiConfig(
                    use_sahi=False,
                    conf=0.15,
                    slice_height=240
                ),
            ),
        )
        
    lda = LeafsDiseaseAnalyzer()
    img = cv2.imread(r"C:\Users\table\PycharmProjects\MojeCos2\segmentyuumozebedo\images\kantong_sakit_3.png")
    img2 = cv2.imread(r"C:\Users\table\PycharmProjects\MojeCos2\segmentyuumozebedo\images\kantong_sakit_7.png")
    (draw_imgs, gowienkos), proc_time = lda.process_image(images=[img, img2], config=config)
    print(proc_time)

    for draw_img, gowienko in zip(draw_imgs, gowienkos):
        cv2.imshow("res", draw_img)
        cv2.waitKey(0)

        for i, gow in enumerate(gowienko):
            print(gow.leaf_id, gow.conf, gow.class_name, gow.diseased_area_perc)
            cv2.imshow(f"res{i}", gow.img_draw)
            cv2.waitKey(0)
        input()