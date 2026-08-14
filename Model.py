# ==============================================================================
# INTELLIGENT REAL-TIME CRIME DETECTION SYSTEM - 3D RES-SE CNN DETECTOR
# ==============================================================================

import os
import sys

# Configure GPU MIG compatibility environment flags
# Target MIG Device 1 (UUID: MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1) — ~16 GB free VRAM
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

# Use file system sharing strategy to allow multi-worker DataLoader parallel processing
mp.set_sharing_strategy('file_system')
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# ==============================================================================
# GPU & SEED CONFIGURATION
# ==============================================================================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    torch.set_num_threads(12)
    print(f"Using CPU for execution ({torch.get_num_threads()} parallel OpenMP CPU threads).")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    ROOT_DATA = "Dataset/"
    VIOLENCE_DIR = os.path.join(ROOT_DATA, "Violence")
    NON_VIOLENCE_DIR = os.path.join(ROOT_DATA, "NonViolence")

    OUTPUT_DIR = "models1"
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_3dcnn_crime_detector.pth")

    MAX_FRAMES = 24
    TARGET_SIZE = (128, 128)
    BATCH_SIZE = 8
    EPOCHS = 40
    BASE_LR = 3e-4
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05

    # Early Stopping Settings
    EARLY_STOP_PATIENCE   = 8      # Stop if val acc doesn't improve for 8 consecutive epochs
    OVERFIT_GAP_THRESHOLD = 15.0   # Stop if train_acc > val_acc by more than 15%

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv')

# ==============================================================================
# DATASET LOADER & AUGMENTATION
# ==============================================================================
# ============================================================
# FULL VIDEO → MULTIPLE CLIPS DATASET
# ============================================================

def load_all_video_frames(
    video_path,
    target_size=Config.TARGET_SIZE
):
    """
    Read the ENTIRE video.

    Returns:
        frames: numpy array
        shape = (T, H, W, C)
    """

    cap = cv2.VideoCapture(video_path)

    frames = []

    if not cap.isOpened():
        print(f"Warning: Could not open {video_path}")
        return None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(
            frame,
            target_size
        )

        frame = frame.astype(
            np.float32
        ) / 255.0

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    return np.array(frames, dtype=np.float32)


def compute_motion(frames):
    """
    Compute frame-to-frame motion.

    Input:
        (T, H, W, C)

    Output:
        (T, H, W, C)
    """

    if len(frames) <= 1:
        return np.zeros_like(frames)

    motion = np.abs(
        np.diff(frames, axis=0)
    )

    # Repeat first motion frame
    first_motion = motion[0:1]

    motion = np.concatenate(
        [first_motion, motion],
        axis=0
    )

    # Same representation used in the original model
    blended = (
        0.7 * frames +
        0.3 * motion
    )

    return blended


class FullVideoClipDataset(Dataset):

    def __init__(
        self,
        video_paths,
        labels,
        augment=False,
        stride=None
    ):

        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment

        # Number of frames between two clips
        #
        # None = non-overlapping clips
        #
        # Example:
        # 24-frame clip
        # stride = 24
        #
        # For more coverage:
        # stride = 12
        #
        if stride is None:
            self.stride = Config.MAX_FRAMES
        else:
            self.stride = stride

        # Build clip index
        self.clip_index = []

        print("\nBuilding full-video clip index...")

        for video_idx, video_path in enumerate(video_paths):

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(
                    f"Warning: Cannot open {video_path}"
                )
                continue

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            cap.release()

            if total_frames <= 0:
                continue

            clip_size = Config.MAX_FRAMES

            # Generate clips throughout ENTIRE video
            start = 0

            while start < total_frames:

                # Keep only clips that contain frames
                if start < total_frames:

                    self.clip_index.append(
                        (
                            video_idx,
                            start
                        )
                    )

                start += self.stride

        print(
            f"Videos: {len(video_paths)}"
        )

        print(
            f"Total training clips: "
            f"{len(self.clip_index)}"
        )


    def __len__(self):

        return len(self.clip_index)


    def __getitem__(self, idx):

        video_idx, start_frame = self.clip_index[idx]

        video_path = self.video_paths[video_idx]

        label = self.labels[video_idx]

        # ----------------------------------------------------
        # Open video
        # ----------------------------------------------------

        cap = cv2.VideoCapture(video_path)

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame
        )

        frames = []

        while len(frames) < Config.MAX_FRAMES:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(
                frame,
                Config.TARGET_SIZE
            )

            frame = frame.astype(
                np.float32
            ) / 255.0

            frames.append(frame)

        cap.release()

        # ----------------------------------------------------
        # Invalid video
        # ----------------------------------------------------

        if len(frames) == 0:

            frames = np.zeros(
                (
                    Config.MAX_FRAMES,
                    Config.TARGET_SIZE[0],
                    Config.TARGET_SIZE[1],
                    3
                ),
                dtype=np.float32
            )

        else:

            frames = np.array(
                frames,
                dtype=np.float32
            )

            # ------------------------------------------------
            # Pad final clip
            # ------------------------------------------------

            while len(frames) < Config.MAX_FRAMES:

                frames = np.concatenate(
                    [
                        frames,
                        frames[-1:]
                    ],
                    axis=0
                )

        frames = frames[
            :Config.MAX_FRAMES
        ]

        # ----------------------------------------------------
        # AUGMENTATION
        # ----------------------------------------------------

        if self.augment:

            # Horizontal flip
            if random.random() > 0.5:

                frames = np.flip(
                    frames,
                    axis=2
                ).copy()

            # Brightness / contrast
            if random.random() > 0.5:

                alpha = random.uniform(
                    0.85,
                    1.15
                )

                beta = random.uniform(
                    -0.1,
                    0.1
                )

                frames = np.clip(
                    frames * alpha + beta,
                    0.0,
                    1.0
                )

        # ----------------------------------------------------
        # Motion
        # ----------------------------------------------------

        blended = compute_motion(frames)

        # ----------------------------------------------------
        # Convert:
        #
        # (T,H,W,C)
        #
        # →
        #
        # (C,T,H,W)
        # ----------------------------------------------------

        frames_tensor = torch.FloatTensor(
            blended
        ).permute(
            3,
            0,
            1,
            2
        )

        label_tensor = torch.FloatTensor(
            [label]
        )

        return frames_tensor, label_tensor
# ==============================================================================
# STRATIFIED DATASET PREPARATION
# ==============================================================================
def prepare_dataset():
    def get_files(folder):
        return sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ])

    vio_files = get_files(Config.VIOLENCE_DIR)
    non_files = get_files(Config.NON_VIOLENCE_DIR)

    all_paths = vio_files + non_files
    all_labels = [1] * len(vio_files) + [0] * len(non_files)

    # 80% train, 20% temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.20, stratify=all_labels, random_state=SEED
    )

    # Split 20% temp into 10% val and 10% test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED
    )

    print(f"Dataset Split Summary:")
    print(f"  Total Videos      : {len(all_paths)}")
    print(f"  Training Set      : {len(train_paths)} samples")
    print(f"  Validation Set    : {len(val_paths)} samples")
    print(f"  Test Set          : {len(test_paths)} samples")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels

# ==============================================================================
# 3D RES-SE CNN ARCHITECTURE
# ==============================================================================
class SEBlock3D(nn.Module):
    """
    3D Squeeze-and-Excitation Motion Attention Module
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock3D, self).__init__()
        reduced_ch = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_ch),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_ch, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        weight = self.fc(x).view(b, c, 1, 1, 1)
        return x * weight

class ResBlock3D(nn.Module):
    """
    3D Residual Block with BatchNorm, ReLU, and SE Motion Attention
    """
    def __init__(self, in_channels, out_channels, stride=(1, 2, 2)):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock3D(out_channels)

        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + residual)
        return out

class CNN3D_ResSE(nn.Module):
    """
    Lightweight 3D Res-SE CNN optimized for Crime Detection (~2.6M parameters)
    """
    def __init__(self, input_channels=3):
        super(CNN3D_ResSE, self).__init__()
        
        # Entry Conv Block
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        
        # Residual Blocks with Squeeze-and-Excitation
        self.layer1 = ResBlock3D(32, 32, stride=(1, 1, 1))
        self.layer2 = ResBlock3D(32, 64, stride=(1, 2, 2))
        self.layer3 = ResBlock3D(64, 128, stride=(1, 2, 2))
        self.layer4 = ResBlock3D(128, 256, stride=(2, 2, 2))
        
        # Global Pooling & Fully Connected Head
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Backward Compatibility Alias
CNN3D = CNN3D_ResSE

# ==============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ==============================================================================
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(train_loader, desc="Training", unit="batch", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

    return running_loss / total, 100.0 * correct / total

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy().flatten())

    acc = 100.0 * correct / total
    avg_loss = running_loss / total
    return avg_loss, acc, np.array(y_true), np.array(y_pred)

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Dry Run check flag
    if "--dry-run" in sys.argv:
        print("\n--- DRY RUN SANITY CHECK ---")
        model = CNN3D_ResSE().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"3D Res-SE CNN Model Loaded. Total Parameters: {total_params:,}")
        dummy_input = torch.randn(2, 3, Config.MAX_FRAMES, Config.TARGET_SIZE[0], Config.TARGET_SIZE[1]).to(device)
        out = model(dummy_input)
        print(f"Forward pass output shape: {out.shape}")
        print("Dry run completed successfully.")
        return

    # Prepare datasets
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_dataset()

    train_dataset = FullVideoClipDataset(
    train_paths,
    train_labels,
    augment=True,
    stride=12
    )

    val_dataset = FullVideoClipDataset(
        val_paths,
        val_labels,
        augment=False,
        stride=24
    )

    test_dataset = FullVideoClipDataset(
        test_paths,
        test_labels,
        augment=False,
        stride=24
    )

    num_workers = 0

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # Model & Optimization
    model = CNN3D_ResSE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Initialized: 3D Res-SE CNN (Total Parameters: {total_params:,})")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.BASE_LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    no_improve_count = 0     # Tracks consecutive epochs without val improvement
    prev_val_acc    = 0.0    # Tracks previous epoch val acc for divergence check

    print("\nStarting Training Pipeline...")
    print("=" * 65)

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{Config.EPOCHS:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  --> Best Checkpoint Saved! (Val Accuracy: {best_val_acc:.2f}%)")
        else:
            no_improve_count += 1

        # ─── Early Stopping Logic ─────────────────────────────────────────────
        # Overfitting check: train acc >> val acc by more than threshold
        overfit_gap = train_acc - val_acc
        if overfit_gap > Config.OVERFIT_GAP_THRESHOLD and epoch > 15:
            print(f"\n[Early Stop] Severe overfitting detected! Train Acc ({train_acc:.2f}%) >> Val Acc ({val_acc:.2f}%) by {overfit_gap:.2f}%. Stopping.")
            break

        # Patience check: stop only if best validation accuracy hasn't improved for PATIENCE epochs
        if no_improve_count >= Config.EARLY_STOP_PATIENCE:
            print(f"\n[Early Stop] Val accuracy did not improve for {Config.EARLY_STOP_PATIENCE} consecutive epochs (best: {best_val_acc:.2f}%). Stopping.")
            break
        # ─────────────────────────────────────────────────────────────────────

    # Evaluate Best Checkpoint on Test Set
    print("\n" + "=" * 65)
    print("EVALUATING BEST MODEL CHECKPOINT ON TEST SET")
    print("=" * 65)
    
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    
    test_loss, test_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device)
    
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=['Non-Violent', 'Violent'], digits=4))

if __name__ == "__main__":
    main()
