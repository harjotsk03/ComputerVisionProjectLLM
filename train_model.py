from ultralytics import YOLO
import yaml

# Load the dataset path from garbage.yaml
def load_dataset_path(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['path']

def train_model():
    # Load the dataset path from YAML file
    yaml_path = '/Users/harjotsingh/Desktop/repos/ComputerVisionProjectLLM/garbage.yaml'
    dataset_path = load_dataset_path(yaml_path)
    
    # Load the model
    model = YOLO('yolov8n-cls.pt')  # Load the pre-trained YOLOv8 classification model
    print("Starting model training...")
    model.train(data=yaml_path, epochs=100, imgsz=640)  # Train the model
    print("Model training completed.")

if __name__ == '__main__':
    train_model()
