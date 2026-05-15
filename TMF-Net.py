import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import multiprocessing


def get_unified_transform(is_training=False):
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class DendrobiumMorphDataset(Dataset):
    def __init__(self, root_dir, year, transform=None):
        self.root_dir = root_dir
        self.year = year
        self.transform = transform
        year_folder_map = {1: "1Y", 2: "2Y", 3: "3Y"}
        self.year_folder = year_folder_map[year]
        
        self.img_paths = []
        month_dirs = ["tn1", "tn2"]
        
        for month_dir in month_dirs:
            month_path = os.path.join(root_dir, self.year_folder, month_dir)
            for img_name in os.listdir(month_path):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.img_paths.append(os.path.join(month_path, img_name))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.year - 1, dtype=torch.long)
        return image, label

class TrainFixedDendrobiumTimeDataset(Dataset):
    def __init__(self, root_dir, year, transform=None):
        self.root_dir = root_dir
        self.year = year
        self.transform = transform
        year_folder_map = {1: "1Y", 2: "2Y", 3: "3Y"}
        self.year_folder = year_folder_map[year]
        
        self.t1_dir = os.path.join(root_dir, self.year_folder, "tn1")
        self.t1_next_dir = os.path.join(root_dir, self.year_folder, "t1")
        self.t2_dir = os.path.join(root_dir, self.year_folder, "tn2")
        self.t2_next_dir = os.path.join(root_dir, self.year_folder, "t2")
        
        for dir_path in [self.t1_dir, self.t1_next_dir, self.t2_dir, self.t2_next_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Temporal folder not found: {dir_path}")
        
        self.t1_images = [f for f in os.listdir(self.t1_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.t1_next_images = [f for f in os.listdir(self.t1_next_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.t2_images = [f for f in os.listdir(self.t2_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.t2_next_images = [f for f in os.listdir(self.t2_next_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        self.all_pairs = []
        
        for t1_img in self.t1_images:
            t1_next_img = t1_img
            if t1_next_img in self.t1_next_images:
                self.all_pairs.append((os.path.join(self.t1_dir, t1_img), 
                                      os.path.join(self.t1_next_dir, t1_next_img)))
        
        for t2_img in self.t2_images:
            t2_next_img = t2_img
            if t2_next_img in self.t2_next_images:
                self.all_pairs.append((os.path.join(self.t2_dir, t2_img), 
                                      os.path.join(self.t2_next_dir, t2_next_img)))
        
        dec_t1_pairs = len([p for p in self.all_pairs if self.t1_dir in p[0]])
        apr_t2_pairs = len(self.all_pairs) - dec_t1_pairs
        
        
    
    def get_statistics(self):
        dec_t1_pairs = len([p for p in self.all_pairs if self.t1_dir in p[0]])
        apr_t2_pairs = len(self.all_pairs) - dec_t1_pairs
        return {
            'year': self.year,
            'year_folder': self.year_folder,
            'dec_count': len(self.t1_images),
            'dec_t1_pairs': dec_t1_pairs,
            'apr_count': len(self.t2_images),
            'apr_t2_pairs': apr_t2_pairs,
            'total_pairs': len(self.all_pairs)
        }

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        t_prev_path, t_next_path = self.all_pairs[idx]
        
        t_prev_img = Image.open(t_prev_path).convert('RGB')
        t_next_img = Image.open(t_next_path).convert('RGB')
        
        if self.transform:
            t_prev_img = self.transform(t_prev_img)
            t_next_img = self.transform(t_next_img)
        
        label = torch.tensor(self.year - 1, dtype=torch.long)
        return t_prev_img, t_next_img, label

class FixedDendrobiumTimeDataset(Dataset):
    def __init__(self, root_dir, year, transform=None):
        self.root_dir = root_dir
        self.year = year
        self.transform = transform
        year_folder_map = {1: "1Y", 2: "2Y", 3: "3Y"}
        self.year_folder = year_folder_map[year]
        
        self.t1_dir = os.path.join(root_dir, self.year_folder, "T1")
        self.t2_dir = os.path.join(root_dir, self.year_folder, "T2")
        
        for dir_path in [self.t1_dir, self.t2_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Temporal folder not found: {dir_path}")
        
        self.t1_images = [f for f in os.listdir(self.t1_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.t2_images = [f for f in os.listdir(self.t2_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        self.all_pairs = []
        
        min_len = min(len(self.t1_images), len(self.t2_images))
        for i in range(min_len):
            t1_img = self.t1_images[i]
            t2_img = self.t2_images[i]
            self.all_pairs.append((os.path.join(self.t1_dir, t1_img), 
                                  os.path.join(self.t2_dir, t2_img)))
        
        dec_t1_pairs = len(self.all_pairs) // 2
        apr_t2_pairs = len(self.all_pairs) - dec_t1_pairs
        
        

    def get_statistics(self):
        return {
            'year': self.year,
            'year_folder': self.year_folder,
            'dec_count': len(self.t1_images),
            'dec_t1_pairs': len(self.all_pairs),
            'apr_count': len(self.t2_images),
            'apr_t2_pairs': len(self.all_pairs),
            'total_pairs': len(self.all_pairs)
        }

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        t_prev_path, t_next_path = self.all_pairs[idx]
        
        t_prev_img = Image.open(t_prev_path).convert('RGB')
        t_next_img = Image.open(t_next_path).convert('RGB')
        
        if self.transform:
            t_prev_img = self.transform(t_prev_img)
            t_next_img = self.transform(t_next_img)
        
        label = torch.tensor(self.year - 1, dtype=torch.long)
        return t_prev_img, t_next_img, label

def build_dual_branch_datasets(root_dir, train_transform, val_transform):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    branch1_datasets = []
    for year in [1, 2, 3]:
        ds = TrainFixedDendrobiumTimeDataset(
            root_dir=root_dir, 
            year=year, 
            transform=train_transform
        )
        branch1_datasets.append(ds)
    
    branch1_dataset = ConcatDataset(branch1_datasets)

    branch2_datasets = []
    for year in [1, 2, 3]:
        ds = DendrobiumMorphDataset(
            root_dir=root_dir, 
            year=year, 
            transform=train_transform
        )
        branch2_datasets.append(ds)
    
    branch2_dataset = ConcatDataset(branch2_datasets)

    val_root_dir = r"D:\dataset\DNS\val"
    
    branch1_val_datasets = []
    for year in [1, 2, 3]:
        ds = FixedDendrobiumTimeDataset(
            root_dir=val_root_dir, 
            year=year, 
            transform=val_transform
        )
        branch1_val_datasets.append(ds)
    
    branch1_val = ConcatDataset(branch1_val_datasets)

    return (branch1_dataset, branch2_dataset), (branch1_val, None)

class DualBranchLoader(Dataset):
    def __init__(self, branch1_ds, branch2_ds, allow_cycle_branch2=False, is_validation=False, is_training=False):
        self.branch1_ds = branch1_ds
        self.branch2_ds = branch2_ds
        self.allow_cycle_branch2 = allow_cycle_branch2
        self.is_validation = is_validation
        self.is_training = is_training
        
        len1 = len(branch1_ds)
        
        if self.is_validation and self.branch2_ds is None:
            self.dataset_len = len1
            return
        
        len2 = len(branch2_ds)
        diff_ratio = abs(len1 - len2) / max(len1, len2)
        
        if self.allow_cycle_branch2:
            self.dataset_len = len1
        else:
            self.dataset_len = min(len1, len2)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        b1_t1, b1_t2, b1_label = self.branch1_ds[idx]
        
        if self.is_training:
            b2_img = b1_t1
            b2_label = b1_label
        elif self.is_validation:
            import random
            b2_img = random.choice([b1_t1, b1_t2])
            b2_label = b1_label
        else:
            if self.branch2_ds is None:
                raise ValueError("branch2_ds cannot be None except in validation mode")
            b2_idx = idx % len(self.branch2_ds) if self.allow_cycle_branch2 else idx
            b2_img, b2_label = self.branch2_ds[b2_idx]
        
        return b1_t1, b1_t2, b1_label, b2_img, b2_label

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetBackbone(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3, pretrained=True, use_cbam=False):
        super(ResNetBackbone, self).__init__()
        self.in_channels = 64
        self.pretrained = pretrained
        self.use_cbam = use_cbam
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.feature_dim = 512
        if hasattr(block, 'expansion'):
            self.feature_dim = 512 * block.expansion
        
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        if self.pretrained:
            self._load_pretrained_weights(block)
    
    def _load_pretrained_weights(self, block):
        pass

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            if hasattr(block, 'expansion'):
                self.in_channels = out_channels * block.expansion
            else:
                self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, cbam_modules=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        
        if self.use_cbam and cbam_modules is not None and len(cbam_modules) > 0:
            out = cbam_modules[0](out)
            
        out = self.layer2(out)
        
        if self.use_cbam and cbam_modules is not None and len(cbam_modules) > 1:
            out = cbam_modules[1](out)
            
        out = self.layer3(out)
        
        if self.use_cbam and cbam_modules is not None and len(cbam_modules) > 2:
            out = cbam_modules[2](out)
            
        out = self.layer4(out)
        
        if self.use_cbam and cbam_modules is not None and len(cbam_modules) > 3:
            out = cbam_modules[3](out)
            
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return out

def ResNet18(num_classes=3, pretrained=True):
    return ResNetBackbone(ResidualBlock, [2, 2, 2, 2], num_classes, pretrained)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        ca_out = self.channel_attention(x) * x
        sa_out = self.spatial_attention(ca_out) * ca_out
        return sa_out

def ResNet50(num_classes=3, pretrained=True, use_cbam=False):
    return ResNetBackbone(Bottleneck, [3, 4, 6, 3], num_classes, pretrained, use_cbam)

class TimeBranch(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(TimeBranch, self).__init__()
        self.resnet = ResNet18(pretrained=pretrained)
        self.feature_dim = self.resnet.feature_dim
        
        self.temporal_attention = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.diff_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, self.feature_dim)
        )
        
        self.feature_norm = nn.LayerNorm(2 * self.feature_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(2 * self.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, t1_img, t2_img):
        feat1 = self.resnet(t1_img)
        feat2 = self.resnet(t2_img)
        
        combined_feat = torch.cat([feat1, feat2], dim=1)
        attention_weight = self.temporal_attention(combined_feat)
        attended_feat = attention_weight * feat1 + (1 - attention_weight) * feat2
        
        feat_diff = torch.abs(feat1 - feat2)
        diff_feat = self.diff_mlp(feat_diff)
        
        fused_feat = torch.cat([attended_feat, diff_feat], dim=1)
        
        normalized_feat = self.feature_norm(fused_feat)
        
        self.fused_feat = normalized_feat
        
        return self.fc(normalized_feat)

class MorphBranch(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(MorphBranch, self).__init__()
        self.resnet = ResNet50(pretrained=pretrained, use_cbam=True)
        self.feature_dim = self.resnet.feature_dim
        
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)
        
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.feature_norm = nn.LayerNorm(1024)
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, img):
        feat = self.resnet(img, [self.cbam1, self.cbam2, self.cbam3, self.cbam4])
        
        projected_feat = self.projection(feat)
        
        normalized_feat = self.feature_norm(projected_feat)
        
        self.resnet_feat = normalized_feat
        
        return self.fc(feat)

class DualBranchResNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(DualBranchResNet, self).__init__()
        self.branch1 = TimeBranch(num_classes=num_classes, pretrained=pretrained)
        self.branch2 = MorphBranch(num_classes=num_classes, pretrained=pretrained)
        
        self.attention_fusion = nn.Sequential(
            nn.Linear(self.branch1.feature_dim * 2 + 1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        
        self.final_fc = nn.Sequential(
            nn.Linear(self.branch1.feature_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, b1_t1, b1_t2, b2_img):
        out1 = self.branch1(b1_t1, b1_t2)
        branch1_feat = self.branch1.fused_feat
        
        out2 = self.branch2(b2_img)
        branch2_feat = self.branch2.resnet_feat
        
        combined_feat = torch.cat([branch1_feat, branch2_feat], dim=1)
        attention_weights = self.attention_fusion(combined_feat)
        
        fused_feat = attention_weights[:, 0].unsqueeze(1) * branch1_feat + \
                      attention_weights[:, 1].unsqueeze(1) * branch2_feat
        
        final_out = self.final_fc(fused_feat)
        return final_out

def setup_multiprocessing():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    try:
        import cv2
        cv2.setNumThreads(8)
    except ImportError:
        pass
    
    torch.set_num_threads(8)

if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    
    ROOT_DIR = r"D:\dataset\DNS\train"
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    
    setup_multiprocessing()
    
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available! Please ensure NVIDIA GPU and CUDA drivers are installed.")
    
    DEVICE = torch.device("cuda:0")

    train_transform = get_unified_transform(is_training=True)
    val_transform = get_unified_transform(is_training=False)
    (branch1_train, branch2_train), (branch1_val, branch2_val) = build_dual_branch_datasets(ROOT_DIR, train_transform, val_transform)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    torch.cuda.empty_cache()
    torch.cuda.memory_allocated()
    
    net = DualBranchResNet(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    
    optimizer = torch.optim.Adam(
        net.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=5e-4
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=8,
        threshold=0.001,
        threshold_mode='rel',
        cooldown=5,
        min_lr=5e-7
    )
    
    from torch.amp import GradScaler, autocast
    scaler = GradScaler()
    
    train_dataset = DualBranchLoader(branch1_train, branch2_train, allow_cycle_branch2=True, is_training=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=6,
        pin_memory=True
    )
    
    val_dataset = DualBranchLoader(branch1_val, branch2_val, allow_cycle_branch2=True, is_validation=True)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_log.txt')
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write("Epoch,Train Accuracy,Train Loss,Validation Accuracy,Validation Loss\n")
    
    best_val_acc = 0.0
    best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resnet-18+50.pth')
    best_epoch = 0
    
    patience = 1000
    counter = 0

    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        net.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (b1_t1, b1_t2, b1_label, b2_img, b2_label) in enumerate(train_dataloader):
            b1_t1, b1_t2, label = b1_t1.to(DEVICE, non_blocking=True), b1_t2.to(DEVICE, non_blocking=True), b1_label.to(DEVICE, non_blocking=True)
            b2_img = b2_img.to(DEVICE, non_blocking=True)

            if label.size(0) == 0:
                continue

            optimizer.zero_grad()
            
            with autocast(device_type="cuda"):
                fused_out = net(b1_t1, b1_t2, b2_img)
                target = label.squeeze()
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                elif fused_out.size(0) != target.size(0):
                    continue
                loss = criterion(fused_out, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_running_loss += loss.item() * b1_t1.size(0)
            _, predicted = torch.max(fused_out.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            train_epoch_loss = train_running_loss / train_total
            train_epoch_acc = train_correct / train_total

        history['train_loss'].append(train_epoch_loss)
        history['train_acc'].append(train_epoch_acc)

        net.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for b1_t1, b1_t2, b1_label, b2_img, b2_label in val_dataloader:
                b1_t1, b1_t2, label = b1_t1.to(DEVICE, non_blocking=True), b1_t2.to(DEVICE, non_blocking=True), b1_label.to(DEVICE, non_blocking=True)
                b2_img = b2_img.to(DEVICE, non_blocking=True)

                if label.size(0) == 0:
                    continue
                
                with autocast(device_type="cuda"):
                    fused_out = net(b1_t1, b1_t2, b2_img)
                    target = label.squeeze()
                    if target.dim() == 0:
                        target = target.unsqueeze(0)
                    elif fused_out.size(0) != target.size(0):
                        continue
                    
                    loss = criterion(fused_out, target)

                    val_running_loss += loss.item() * b1_t1.size(0)
                    _, predicted = torch.max(fused_out.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    label_np = target.cpu().numpy()
                    if label_np.ndim == 0:
                        all_labels.append(label_np.item())
                    else:
                        all_labels.extend(label_np)
                    all_probs.extend(F.softmax(fused_out, dim=1).cpu().numpy())
                
                if val_total > 0:
                    current_val_acc = val_correct / val_total
                else:
                    current_val_acc = 0.0
                val_loss_value = val_running_loss/(val_total if val_total > 0 else 1)
                val_progress.set_postfix({'val_loss': f'{val_loss_value:.4f}', 'val_acc': f'{current_val_acc:.4f}'})
            
            if val_total > 0:
                val_epoch_loss = val_running_loss / val_total
                val_epoch_acc = val_correct / val_total
            else:
                val_epoch_loss = 0.0
                val_epoch_acc = 0.0
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{epoch+1},{train_epoch_acc:.4f},{train_epoch_loss:.4f},{val_epoch_acc:.4f},{val_epoch_loss:.4f}\n")

        if len(all_labels) > 0:
            class_names = ['1Y', '2Y', '3Y']
            report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
            overall_f1 = report['macro avg']['f1-score']

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_epoch = epoch + 1
            counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_epoch_loss,
                'val_loss': val_epoch_loss,
                'train_acc': train_epoch_acc,
                'val_acc': val_epoch_acc,
                'best_val_acc': best_val_acc
            }, best_model_path)
        else:
            counter += 1
        
        if len(all_labels) > 0:
            scheduler.step(overall_f1)
        else:
            raise RuntimeError(f"Epoch {epoch+1}: No valid validation data! Please check if validation set is empty or data loading is correct.")