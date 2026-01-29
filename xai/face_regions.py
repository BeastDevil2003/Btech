import cv2
import numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]
MOUTH = [78, 308, 13, 14]
NOSE = [1, 2, 98]

def get_face_regions(image):
    h, w, _ = image.shape
    results = mp_face.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    mask = np.zeros((h, w), dtype=np.uint8)
    regions = {}

    if not results.multi_face_landmarks:
        return mask, regions

    lm = results.multi_face_landmarks[0].landmark

    def draw_region(indices, name):
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
        cv2.fillPoly(mask, [np.array(pts)], 255)
        regions[name] = pts

    draw_region(LEFT_EYE, "left_eye")
    draw_region(RIGHT_EYE, "right_eye")
    draw_region(MOUTH, "mouth")
    draw_region(NOSE, "nose")

    return mask, regions
