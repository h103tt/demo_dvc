from ultralytics import YOLO
from clearml import Task
import subprocess

print("Pulling dataset from DVC...")
subprocess.run(["dvc", "pull"], check=True)
#wait here
#check
task = Task.init(project_name = 'yolov8', task_name='training' )
# Log the dataset tracker file to the ClearML experiment
task.upload_artifact(name='dataset_dvc_pointer', artifact_object='demo_dataset/data.dvc')

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="demo_dataset/data/data.yaml", epochs=2, imgsz=320, batch=4, workers=0)

