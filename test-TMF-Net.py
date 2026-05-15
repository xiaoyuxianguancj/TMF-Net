import os
import sys
import torch
from torch.utils.data import DataLoader
from PIL import Image
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import importlib.util
    module_path = r"C:\Users\30393\Desktop\下一篇论文准备\TMF-Net\TMF-Net.py"
    spec = importlib.util.spec_from_file_location("branch_resnet_model", module_path)
    branch_resnet_model = importlib.util.module_from_spec(spec)
    sys.modules["branch_resnet_model"] = branch_resnet_model
    spec.loader.exec_module(branch_resnet_model)
    
    DualBranchResNet = branch_resnet_model.DualBranchResNet
    get_unified_transform = branch_resnet_model.get_unified_transform
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
except Exception as e:
    print(f"Error importing model: {e}")
    sys.exit(1)

TEST_DATA_PATH = r"D:\dataset\DNS\test"
MODEL_PATH = r"C:\Users\30393\Desktop\下一篇论文准备\TMF-Net\TMF-Net.pth"
BATCH_SIZE = 8
NUM_CLASSES = 3

def get_test_transform():
    return get_unified_transform(is_training=False)

class DendrobiumTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.year_folders = {0: "1Y", 1: "2Y", 2: "3Y"}
        self.data_pairs = []
        
        for class_idx, class_name in self.year_folders.items():
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                t1_path = os.path.join(class_path, "T1")
                t2_path = os.path.join(class_path, "T2")
                if os.path.exists(t1_path) and os.path.exists(t2_path):
                    t1_images = [os.path.join(t1_path, f) for f in os.listdir(t1_path) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    t2_images = [os.path.join(t2_path, f) for f in os.listdir(t2_path) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    
                    min_length = min(len(t1_images), len(t2_images))
                    for i in range(min_length):
                        self.data_pairs.append((t1_images[i], t2_images[i], t1_images[i], class_idx))
        
        if len(self.data_pairs) > 1000:
            self.data_pairs = random.sample(self.data_pairs, 1000)
        
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        img_path1, img_path2, img_path3, label = self.data_pairs[idx]
        image1 = Image.open(img_path1).convert('RGB')
        image2 = Image.open(img_path2).convert('RGB')
        image3 = Image.open(img_path3).convert('RGB')
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)
        
        return image1, image2, image3, label, img_path1, img_path2

def load_model(model_path, num_classes=NUM_CLASSES):
    print(f"Loading model from: {model_path}")
    model = DualBranchResNet(num_classes=num_classes)
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        print("Checkpoint loaded successfully")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Using model_state_dict from checkpoint")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")
        
        for key in list(state_dict.keys()):
            if key.startswith('module.'):
                state_dict[key[7:]] = state_dict.pop(key)
        
        model.load_state_dict(state_dict, strict=True)
        print("Model weights loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    model.eval()
    return model

def test_model(model, test_loader, device):
    print("Starting test...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (b1_t1, b1_t2, b2_img, labels, _, _) in enumerate(test_loader):
            print(f"Processing batch {batch_idx+1}")
            b1_t1 = b1_t1.to(device)
            b1_t2 = b1_t2.to(device)
            b2_img = b2_img.to(device)
            labels = labels.to(device)
            
            outputs = model(b1_t1, b1_t2, b2_img)
            _, predicted = torch.max(outputs.data, 1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy

if __name__ == "__main__":
    print(f"Test data path: {TEST_DATA_PATH}")
    print(f"Model path: {MODEL_PATH}")
    
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Test data path not found: {TEST_DATA_PATH}")
        sys.exit(1)
    print("Test data path exists")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model path not found: {MODEL_PATH}")
        sys.exit(1)
    print("Model path exists")
    
    print("Creating test transform...")
    test_transform = get_test_transform()
    
    print("Creating test dataset...")
    test_dataset = DendrobiumTestDataset(TEST_DATA_PATH, transform=test_transform)
    
    if len(test_dataset) == 0:
        print("No images found in test set")
        sys.exit(1)
    print(f"Test dataset size: {len(test_dataset)}")
    
    print("Creating test loader...")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("Loading model...")
    model = load_model(MODEL_PATH).to(device)
    
    print("Testing model...")
    test_model(model, test_loader, device)    
    print("Test completed!")
