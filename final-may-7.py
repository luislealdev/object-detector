import torch
import cv2
import os
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.model = YOLO("yolov8n.pt")  # Asegúrate de tener el modelo correcto en esta ruta
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detected_classes = set()

    def predict(self, im0):
        """Run prediction using a YOLO model for the input image `im0`."""
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        self.end_time = time()
        fps = 1 / (self.end_time - self.start_time)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
            # Announce the class name of detected objects if it's new
            if cls not in self.detected_classes:
                os.system(f"say 'Detected {names[int(cls)]}'")
                self.detected_classes.add(cls)
        return im0, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)
            self.display_fps(im0)
            cv2.imshow('YOLOv8 Detection', im0)
            if cv2.waitKey(1) == ord('q'):  # Puedes cerrar la ventana presionando 'q'
                break
        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()
