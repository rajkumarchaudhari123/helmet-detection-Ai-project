from roboflow import Roboflow
from config import ROBOFLOW_API_KEY

print("Connecting to Roboflow...")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

print("Finding helmet dataset...")
project = rf.workspace("roboflow-100").project("hard-hat-detection")

print("Downloading... (ye 2-3 minute lagega)")
dataset = project.version(2).download("yolov8")

print("✅ Done!")
print(f"Folder name: {dataset.location}")