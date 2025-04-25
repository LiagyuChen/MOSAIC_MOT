import openvino as ov
from nncf import quantize, QuantizationPreset
from ultralytics.data import load_inference_source  # fast dataloader helper

calib_data = load_inference_source("demo/toy1_frames", batch=1, vid_stride=32)

core   = ov.Core()
ov_dir = "weights/yoloe-11l-seg-pf_openvino_model/yoloe-11l-seg-pf.xml"
model  = core.read_model(ov_dir)

quant_params = {
    "preset": QuantizationPreset.MIXED,
    "target_device": "CPU",
    "subset_size": 300
}

q_model = quantize(model, calib_data, **quant_params)   # NNCF PTQ
int8_path = "weights/yoloe-11l-seg-pf_openvino_model_int8"
int8_path.mkdir(exist_ok=True)
ov.serialize(q_model, int8_path / "model.xml")
print("INT8 model stored in", int8_path)
