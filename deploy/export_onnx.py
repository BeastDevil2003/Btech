import torch
from model.model import DeepFakeDetectionModel

model = DeepFakeDetectionModel()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    "deepfake_detector.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print(" ONNX model exported")
