import os
import yaml

def check_dataset_structure(dataset_path):
    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return
    
    # Directories for train and val
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    
    if not os.path.exists(train_path):
        print(f"Error: Train directory '{train_path}' does not exist.")
        return
    
    if not os.path.exists(val_path):
        print(f"Error: Validation directory '{val_path}' does not exist.")
        return
    
    # Check subdirectories in train
    class_dirs = ['recycling', 'compost', 'landfill']
    for class_dir in class_dirs:
        train_class_dir = os.path.join(train_path, class_dir)
        val_class_dir = os.path.join(val_path, class_dir)
        
        if not os.path.exists(train_class_dir):
            print(f"Error: Train class directory '{train_class_dir}' does not exist.")
        elif len(os.listdir(train_class_dir)) == 0:
            print(f"Warning: Train class directory '{train_class_dir}' is empty.")
        
        if not os.path.exists(val_class_dir):
            print(f"Error: Validation class directory '{val_class_dir}' does not exist.")
        elif len(os.listdir(val_class_dir)) == 0:
            print(f"Warning: Validation class directory '{val_class_dir}' is empty.")
    
    print("Dataset structure is correct.")

# Load the dataset path from garbage.yaml
def load_dataset_path(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['path']

# Define the path to your garbage.yaml file
yaml_path = './garbage.yaml'

# Get the dataset path from the YAML file
dataset_path = load_dataset_path(yaml_path)

# Check the dataset structure
check_dataset_structure(dataset_path)
