from ultralytics import YOLOE
from pathlib import Path

# -------- paths --------------------------------------------------------------
ckpt = Path("weights/yoloe-11l-seg-pf.pt")               # your trained model
export_root = Path("openvino_models") / ckpt.stem
export_root.mkdir(parents=True, exist_ok=True)

# -------- export -------------------------------------------------------------
yoloe_model = YOLOE(ckpt)                                 # load PyTorch model
ov_dir = yoloe_model.export(format="openvino",           # invokes Ultralytics exporter
                     half=True,                   # FP16 weights
                     imgsz=640,                   # static image shape speeds up IR
                     device="cpu")               # export on CPU to avoid GPU deps
print(f"FP16 IR saved to {ov_dir}")
