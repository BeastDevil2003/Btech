import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("deepfake_detector.onnx")

dummy = np.random.rand(1, 3, 224, 224).astype(np.float32)
out = sess.run(None, {"input": dummy})

print("Prediction:", np.argmax(out[0]))
