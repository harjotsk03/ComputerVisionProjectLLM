from ultralytics import YOLO
import yaml

# Load the dataset path from garbage.yaml
def load_dataset_path(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['path']

def train_model():
    yaml_path = 'garbage.yaml'
    dataset_path = load_dataset_path(yaml_path)
    
    model = YOLO('yolov8n-cls.pt')
    print("Starting model training...")
    model.train(data=dataset_path, epochs=100, imgsz=640)
    print("Model training completed.")

if __name__ == '__main__':
    train_model()
