from pathlib import Path
from ultralytics import YOLOE


ckpt = Path("weights/yoloe-11l-seg-pf.pt")
export_root = Path("openvino_models") / ckpt.stem
export_root.mkdir(parents=True, exist_ok=True)

yoloe_model = YOLOE(ckpt)
ov_dir = yoloe_model.export(format="openvino",
    half=True,
    imgsz=640,
    device="cpu"
)
print(f"FP16 IR saved to {ov_dir}")
