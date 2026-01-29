import torch
from torchvision import transforms
from PIL import Image

model = torch.jit.load("deepfake_detector_ts.pt")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

img = Image.open("test.jpg").convert("RGB")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(x)
    pred = out.argmax(dim=1).item()

print("Prediction:", "Fake" if pred == 1 else "Real")
