from ultralytics import YOLO
import yaml

def load_dataset_path(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['path']

def train_model():
    yaml_path = 'garbage.yaml'
    dataset_path = load_dataset_path(yaml_path)
    
    model = YOLO('yolov8n-cls.pt')
    print("Starting model training...")
    model.train(data=dataset_path, epochs=5, imgsz=640)
    print("Model training completed.")


def test_model():
    model = YOLO('runs/classify/train2/weights/best.pt')

    # Run inference on the single image
    results = model("trash15.jpg")
    class_names = ['compost', 'landfill', 'recycling']

    # Parse and print the results
    for result in results:
        # Print all class probabilities for debugging
        for idx, prob in enumerate(result.probs.data.tolist()):
            print(f"{class_names[idx]}: {prob:.2f}")

        # Get the index and confidence of the top prediction
        top_class_idx = result.probs.top1  # Index of the top predicted class
        top_confidence = result.probs.top1conf.item()  # Confidence of the top prediction

        # Get the class name from the index
        top_class_name = class_names[top_class_idx]

        # Print the top prediction result
        print(f"Predicted class: {top_class_name} with confidence: {top_confidence:.2f}")


if __name__ == '__main__':
    test_model()
