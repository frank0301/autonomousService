import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import numpy as np
import cv2

_MODEL_ID = "IDEA-Research/grounding-dino-base"
class GroundingDINOInfer:
    def __init__(self, model_id=_MODEL_ID, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def infer(self, img, task):
        # image_resize = cv2.resize(img, (image_cv.shape[1] // 2, image_cv.shape[0] // 2))
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)

        inputs = self.processor(images=image, text=task, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold = 0.4,
            text_threshold = 0.3,
            target_sizes=[image.size[::-1]]
        )

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        boxes = results[0]['boxes'].cpu().numpy()
        labels = results[0]['labels']
        scores = results[0]['scores'].cpu().numpy()

        rect = None
        center = None

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            rect=[x1, y1, x2, y2]
            center=[(x1 + x2) // 2, (y1 + y2) // 2]

            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"{label} {score:.2f}"
            cv2.putText(image_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return image_cv, rect, center
