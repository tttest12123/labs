yolo = """
from ultralytics import YOLO
import shutil
import os
from google.colab import drive

drive.mount('/content/drive')


model = YOLO('yolov8.yaml', task="detect")

results = model.train(
    data='/content/drive/MyDrive/abc/data.yaml',
    epochs=35,
    batch=16,
    imgsz=640,
    save_period=5,
)


drive_path = '/content/drive/MyDrive/yolo_training'
if not os.path.exists(drive_path):
    os.makedirs(drive_path)

run_folder_path = 'runs/detect'
shutil.copytree(run_folder_path, os.path.join(drive_path, 'detect_runs'))

"""