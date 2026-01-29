import torch
from model.model import DeepFakeDetectionModel

DEVICE = "cpu" 

model = DeepFakeDetectionModel()
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

dummy = torch.randn(1, 3, 224, 224)


scripted_model = torch.jit.trace(model, dummy)

scripted_model.save("deepfake_detector_ts.pt")

print(" TorchScript model saved")
