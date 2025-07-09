import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import cv2

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)
image_cv = cv2.imread("/home/epon/vl_test/imgs/image001.jpg")#labSen/frame_pic003.jpg")
image_resize= cv2.resize(image_cv, (image_cv.shape[1] // 2, image_cv.shape[0] // 2))
image_rgb = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image_rgb)
# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
task=input("input a task, end with dot'.', (can be multipe)")

text = task#"a chair. a door."
print(text)
inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

print(results)

import numpy as np


image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

boxes = results[0]['boxes'].cpu().numpy()
labels = results[0]['labels']
scores = results[0]['scores'].cpu().numpy()

for box, label, score in zip(boxes, labels, scores):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
    text = f"{label} {score:.2f}"
    cv2.putText(image_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

if image_cv is not None and image_cv.size != 0:
    cv2.imshow("Detection Result", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Warning: Empty or invalid image received. Skipping display.")
