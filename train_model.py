from ultralytics import YOLO
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def load_dataset_path(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['path']

def train_model():
    yaml_path = 'garbage.yaml'
    dataset_path = load_dataset_path(yaml_path)
    
    model = YOLO('yolov8n-cls.pt')
    print("Starting model training...")
    model.train(data=dataset_path, epochs=5, imgsz=640, batch=16)
    print("Model training completed.")

def test_model():
    model = YOLO('runs/classify/train4/weights/best.pt')

    # Run inference on the single image
    results = model("biological123.jpg")
    class_names = ['compost', 'landfill', 'recycling']

    # Parse and print the results
    for result in results:
        print(result)
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

def evaluate_model():
    # Load the model
    model = YOLO('runs/classify/train4/weights/best.pt')

    # Evaluate on the validation set
    results = model.val()  # This method evaluates on the dataset used during training
    print(results)
    
    # Access metrics directly
    top1_accuracy = results.top1  # Top-1 accuracy
    top5_accuracy = results.top5  # Top-5 accuracy
    results_dict = results.results_dict  # Access the results dictionary
    
    # Assuming results_dict contains precision, recall, and F1 score
    precision = results_dict.get('precision', 0)  # Use get to avoid KeyError if not present
    recall = results_dict.get('recall', 0)
    f1 = results_dict.get('f1', 0)
    accuracy = results_dict.get('accuracy', 0)

    print(f"Top-1 Accuracy: {top1_accuracy:.2f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Accuracy: {accuracy:.2f}")

# Example usage
if __name__ == '__main__':
    train_model()