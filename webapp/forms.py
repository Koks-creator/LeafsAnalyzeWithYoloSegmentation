import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, MultipleFileField
from wtforms import BooleanField, FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange, ValidationError

available_models = [
                    "model_s", "model_m", "model_s2"
                    ]


def max_files_count(max_count: int):
    """Zwraca funkcję-walidator sprawdzającą maksymalną liczbę plików."""
    def _max_files_count(form, field):
        # field.data to lista obiektów FileStorage
        if len(field.data) > max_count:
            raise ValidationError(f"You can upload max {max_count} files.")
    return _max_files_count


class MainForm(FlaskForm):
    images_field = MultipleFileField("Upload files",
        validators=[DataRequired(),
                    FileAllowed(["jpg", "png", "jpeg"]),
                    max_files_count(Config.MAX_IMAGE_FILES) 
                    ]
        )
    filter_lt_px_field = IntegerField(
        label="Filter objects larger than X px",
        validators=[
            DataRequired(),
            NumberRange(min=1000, max=200000),
            ],
            default=50000
        )
    yolo_conf_thr_leaf_field = FloatField(
        label="YOLO confidence threshold Leaf model",
        validators=[
            DataRequired(),
            NumberRange(min=0.1, max=0.9),
            ],
            default=0.2
        )
    yolo_iou_leaf_field = FloatField(
            label="YOLO iou Leaf model",
            validators=[
                DataRequired(),
                NumberRange(min=0.1, max=0.9),
                ],
                default=0.2
            )
    yolo_conf_thr_dis_field = FloatField(
        label="YOLO confidence threshold Disease model",
        validators=[
            DataRequired(),
            NumberRange(min=0.1, max=0.9),
            ],
            default=0.2
        )
    yolo_iou_dis_field = FloatField(
        label="YOLO iou Disease model",
        validators=[
            DataRequired(),
            NumberRange(min=0.1, max=0.9),
            ],
            default=0.2
        )
    use_sahi_leaf_field = BooleanField(
        label="Use SAHI Leaf model",
        default=True
    )
    sahi_conf_thr_leaf_field = FloatField(
        label="SAHI confidence threshold",
        validators=[
            DataRequired(),
            NumberRange(min=0.1, max=0.9),
            ],
            default=0.2
        )
    sahi_slice_height_leaf_field = IntegerField(
        label="SAHI slice height Leaf model",
        validators=[
            DataRequired(),
            NumberRange(min=120, max=480),
            ],
            default=240
        )
    sahi_slice_width_leaf_field = IntegerField(
        label="SAHI slice width Leaf model",
        validators=[
            DataRequired(),
            NumberRange(min=120, max=480),
            ],
            default=240
        )
    sahi_overlap_height_ratio_leaf_field = FloatField(
        label="SAHI overlap height ratio Leaf model",
        validators=[
            DataRequired(),
            NumberRange(min=0.1, max=0.9),
            ],
            default=0.3
        )
    sahi_overlap_width_ratio_leaf_field = FloatField(
        label="SAHI overlap width ratio Leaf model",
        validators=[
            DataRequired(),
            NumberRange(min=0.1, max=0.9),
            ],
            default=0.3
        )
    sahi_match_thr_leaf_field = FloatField(
            label="SAHI match threshold Leaf model",
            validators=[
                DataRequired(),
                NumberRange(min=0.1, max=0.9),
                ],
                default=0.4
            )
    use_sahi_dis_field = BooleanField(
            label="Use SAHI Disease model",
            default=True
        )
    sahi_conf_thr_dis_field = FloatField(
        label="SAHI confidence threshold Disease model",
        validators=[
            DataRequired(),
            NumberRange(min=0.1, max=0.9),
            ],
            default=0.2
        )
    sahi_slice_height_dis_field = IntegerField(
        label="SAHI slice height Disease model",
        validators=[
            DataRequired(),
            NumberRange(min=120, max=480),
            ],
            default=240
        )
    sahi_slice_width_dis_field = IntegerField(
        label="SAHI slice width Disease model",
        validators=[
            DataRequired(),
            NumberRange(min=120, max=480),
            ],
            default=240
        )
    sahi_overlap_height_ratio_dis_field = FloatField(
        label="SAHI overlap height ratio Disease model",
        validators=[
            DataRequired(),
            NumberRange(min=0.1, max=0.9),
            ],
            default=0.3
        )
    sahi_overlap_width_ratio_dis_field = FloatField(
        label="SAHI overlap width ratio Disease model",
        validators=[
            DataRequired(),
            NumberRange(min=0.1, max=0.9),
            ],
            default=0.3
        )
    sahi_match_thr_dis_field = FloatField(
            label="SAHI match threshold Disease model",
            validators=[
                DataRequired(),
                NumberRange(min=0.1, max=0.9),
                ],
                default=0.4
            )
    models_list_field = SelectField("Select model",
                                    choices=[(m, m) for m in available_models])
    submit_field = SubmitField("Submit")
