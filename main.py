from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')

results = model.predict(source="0", show=True ,verbose=False)  # accepts all for
for r in results:
    print(r)

# result = 'holaaaaaaaa'

# os.system(f"say '{result}'")
