# ==============================================================================
# INTELLIGENT REAL-TIME CRIME DETECTION SYSTEM - 3D RES-SE CNN DETECTOR
# Phase 1: Spatiotemporal Binary Crime vs. No-Crime Classifier
# ==============================================================================

import os
import sys
import json
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
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from tqdm import tqdm

# ==============================================================================
# GPU & ENVIRONMENT CONFIGURATION
# ==============================================================================
# Configure GPU environment flags (preserve existing environment variable if already configured)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1"

os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ==============================================================================
# HARDWARE SELECTION & DIAGNOSTICS
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n" + "=" * 70)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[HARDWARE] GPU DETECTED: Using '{gpu_name}'")
    print(f"   - Active GPU Count  : {gpu_count}")
    print(f"   - Dedicated VRAM    : {vram_gb:.2f} GB")
    print(f"   - CUDA Driver / Ver : {torch.version.cuda}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
else:
    cpu_threads = min(12, os.cpu_count() or 4)
    torch.set_num_threads(cpu_threads)
    print(f"[HARDWARE] GPU NOT DETECTED: Falling back to CPU execution.")
    print(f"   - Allocated Cores   : {cpu_threads} parallel OpenMP CPU threads")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"   - Filter Active     : CUDA_VISIBLE_DEVICES='{os.environ['CUDA_VISIBLE_DEVICES']}'")
print("=" * 70 + "\n")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv')


def resolve_dataset_dirs(root_data="Dataset"):
    """
    Exclusively targets and resolves paths for the Real Life Violence Dataset:
      - Violence/ (1,000 clips)
      - NonViolence/ (1,000 clips)
    """
    candidates = [
        os.path.join(root_data, "real-life-violence-situations-dataset", "real life violence situations", "Real Life Violence Dataset"),
        os.path.join(root_data, "Real Life Violence Dataset"),
        os.path.join(root_data, "Violence"),
    ]

    for base in candidates:
        if os.path.basename(base) == "Violence":
            v_dir = base
            nv_dir = os.path.join(os.path.dirname(base), "NonViolence")
        else:
            v_dir = os.path.join(base, "Violence")
            nv_dir = os.path.join(base, "NonViolence")

        if os.path.isdir(v_dir) and os.path.isdir(nv_dir):
            return v_dir, nv_dir

    # Default direct path to Real Life Violence Dataset
    default_base = os.path.join(
        root_data,
        "real-life-violence-situations-dataset",
        "real life violence situations",
        "Real Life Violence Dataset"
    )
    return os.path.join(default_base, "Violence"), os.path.join(default_base, "NonViolence")


class Config:
    ROOT_DATA = "Dataset"
    
    # Exclusively configure Real Life Violence Dataset
    VIOLENCE_DIR, NON_VIOLENCE_DIR = resolve_dataset_dirs(ROOT_DATA)

    OUTPUT_DIR = "models1"
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_3dcnn_crime_detector.pth")
    FULL_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_checkpoint_full.pth")
    METRICS_LOG_PATH = os.path.join(OUTPUT_DIR, "training_metrics.json")
    EVAL_REPORT_PATH = os.path.join(OUTPUT_DIR, "evaluation_report.json")

    MAX_FRAMES = 24
    TARGET_SIZE = (128, 128)
    BATCH_SIZE = 8
    EPOCHS = 40
    BASE_LR = 3e-4
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 3
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05
    CLIP_GRAD_NORM = 1.0

    # Early Stopping & Checkpointing Settings
    EARLY_STOP_PATIENCE   = 10     # Stop if validation metric does not improve for 10 consecutive epochs
    OVERFIT_GAP_THRESHOLD = 20.0   # Stop if train_acc > val_acc by more than 20% after epoch 20

# ==============================================================================
# DATASET LOADER, MOTION PREPROCESSING & AUGMENTATION
# ==============================================================================

def compute_motion(frames):
    """
    Compute frame-to-frame temporal motion via differencing and blend with RGB.
    Dynamic motion contrast is boosted so kinetic cues (punches, swings, kicks)
    stand out clearly from static background pixels.

    Input:
        frames: numpy array shape (T, H, W, C) in [0.0, 1.0]

    Output:
        blended: numpy array shape (T, H, W, C) in [0.0, 1.0]
    """
    if len(frames) <= 1:
        return frames

    motion = np.abs(np.diff(frames, axis=0))
    first_motion = motion[0:1]
    motion = np.concatenate([first_motion, motion], axis=0)

    # Contrast enhancement on motion vector differences
    enhanced_motion = np.clip(motion * 1.5, 0.0, 1.0)

    # Spatiotemporal blend (70% RGB context, 30% dynamic kinetic motion)
    blended = 0.7 * frames + 0.3 * enhanced_motion
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


class FullVideoClipDataset(Dataset):
    """
    Efficient video clip dataset that generates sliding window clips across videos.
    Supports spatiotemporal data augmentation (horizontal flip, color jitter,
    spatial cutout, and speed variation) to ensure robust generalization.
    """
    def __init__(
        self,
        video_paths,
        labels,
        augment=False,
        stride=None,
        target_size=Config.TARGET_SIZE,
        max_frames=Config.MAX_FRAMES
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.augment = augment
        self.target_size = target_size
        self.max_frames = max_frames
        self.stride = stride if stride is not None else max_frames

        # Build clip index: list of tuples (video_idx, start_frame)
        self.clip_index = []
        print("\nIndexing video clips...")

        for video_idx, video_path in enumerate(video_paths):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Cannot open video file: {video_path}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total_frames <= 0:
                continue

            start = 0
            while start < total_frames:
                self.clip_index.append((video_idx, start))
                start += self.stride

        print(f"  Videos: {len(video_paths)} | Total Clips: {len(self.clip_index)}")

    def __len__(self):
        return len(self.clip_index)

    def __getitem__(self, idx):
        video_idx, start_frame = self.clip_index[idx]
        video_path = self.video_paths[video_idx]
        label = float(self.labels[video_idx])

        # Read clip frames
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        raw_frames = []
        # Temporal jitter during augmentation: occasional 2x step sampling
        step = random.choice([1, 2]) if (self.augment and random.random() < 0.25) else 1

        frame_count = 0
        while len(raw_frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % step == 0:
                frame = cv2.resize(frame, self.target_size)
                raw_frames.append(frame.astype(np.float32) / 255.0)
            frame_count += 1

        cap.release()

        # Handle empty/corrupt clips
        if len(raw_frames) == 0:
            frames = np.zeros(
                (self.max_frames, self.target_size[0], self.target_size[1], 3),
                dtype=np.float32
            )
        else:
            frames = np.array(raw_frames, dtype=np.float32)
            # Pad short clips by repeating the final frame
            if len(frames) < self.max_frames:
                pad_count = self.max_frames - len(frames)
                last_frame = np.repeat(frames[-1:], pad_count, axis=0)
                frames = np.concatenate([frames, last_frame], axis=0)

        frames = frames[:self.max_frames]

        # ----------------------------------------------------------------------
        # SPATIOTEMPORAL DATA AUGMENTATIONS (Training only)
        # ----------------------------------------------------------------------
        if self.augment:
            # 1. Random Horizontal Flip (spatially consistent across all frames)
            if random.random() > 0.5:
                frames = np.flip(frames, axis=2).copy()

            # 2. Random Brightness & Contrast Perturbation
            if random.random() > 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-0.08, 0.08)
                frames = np.clip(frames * alpha + beta, 0.0, 1.0)

            # 3. Random Spatial Cutout / Patch Occlusion (simulates CCTV occlusions)
            if random.random() < 0.30:
                h, w = self.target_size
                cut_h = random.randint(12, h // 4)
                cut_w = random.randint(12, w // 4)
                y1 = random.randint(0, h - cut_h)
                x1 = random.randint(0, w - cut_w)
                frames[:, y1:y1 + cut_h, x1:x1 + cut_w, :] = 0.0

        # ----------------------------------------------------------------------
        # MOTION FEATURE COMPUTATION
        # ----------------------------------------------------------------------
        blended = compute_motion(frames)

        # Convert (T, H, W, C) -> (C, T, H, W) for PyTorch 3D Convolution
        frames_tensor = torch.FloatTensor(blended).permute(3, 0, 1, 2)
        label_tensor = torch.FloatTensor([label])

        return frames_tensor, label_tensor, video_idx


# ==============================================================================
# STRATIFIED DATASET PREPARATION
# ==============================================================================
def prepare_dataset():
    """
    Scans video directories and performs an 80/10/10 stratified split by video files
    to prevent data leakage between train, validation, and test sets.
    """
    # Dynamically verify and resolve directories
    vio_dir, non_dir = resolve_dataset_dirs(Config.ROOT_DATA)
    if vio_dir:
        Config.VIOLENCE_DIR = vio_dir
    if non_dir:
        Config.NON_VIOLENCE_DIR = non_dir

    def get_files(folder):
        if not os.path.exists(folder):
            return []
        return sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ])

    vio_files = get_files(Config.VIOLENCE_DIR)
    non_files = get_files(Config.NON_VIOLENCE_DIR)

    if len(vio_files) == 0 or len(non_files) == 0:
        raise FileNotFoundError(
            f"Dataset check failed! Could not find video files in detected directories:\n"
            f"  Violence Dir ({len(vio_files)} videos): {Config.VIOLENCE_DIR}\n"
            f"  Non-Violence Dir ({len(non_files)} videos): {Config.NON_VIOLENCE_DIR}\n"
            f"Please ensure videos (.mp4, .avi, etc.) exist under '{Config.ROOT_DATA}'."
        )

    print(f"Resolved Dataset Folders:")
    print(f"  Violence Videos     : {len(vio_files)} files in '{Config.VIOLENCE_DIR}'")
    print(f"  Non-Violence Videos : {len(non_files)} files in '{Config.NON_VIOLENCE_DIR}'")

    # Create convenience symlinks Dataset/Violence and Dataset/NonViolence if missing
    try:
        sym_vio = os.path.join(Config.ROOT_DATA, "Violence")
        sym_non = os.path.join(Config.ROOT_DATA, "NonViolence")
        if not os.path.exists(sym_vio) and os.path.abspath(sym_vio) != os.path.abspath(Config.VIOLENCE_DIR):
            os.symlink(os.path.abspath(Config.VIOLENCE_DIR), sym_vio)
        if not os.path.exists(sym_non) and os.path.abspath(sym_non) != os.path.abspath(Config.NON_VIOLENCE_DIR):
            os.symlink(os.path.abspath(Config.NON_VIOLENCE_DIR), sym_non)
    except OSError:
        pass

    all_paths = vio_files + non_files
    all_labels = [1] * len(vio_files) + [0] * len(non_files)

    # 80% train, 20% temp (10% val, 10% test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.20, stratify=all_labels, random_state=SEED
    )

    # Split 20% temp into 10% val and 10% test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED
    )

    print("\nDataset Stratified Split Summary:")
    print(f"  Total Videos      : {len(all_paths)} ({len(vio_files)} Violent, {len(non_files)} Non-Violent)")
    print(f"  Training Set      : {len(train_paths)} videos")
    print(f"  Validation Set    : {len(val_paths)} videos")
    print(f"  Test Set          : {len(test_paths)} videos")

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


# ==============================================================================
# 3D RES-SE CNN ARCHITECTURE
# ==============================================================================
class SEBlock3D(nn.Module):
    """
    3D Squeeze-and-Excitation Channel & Motion Attention Module.
    Dynamically recalibrates spatiotemporal feature channels based on motion kinetics.
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
    3D Residual Block with Batch Normalization, ReLU, and SE Motion Attention.
    """
    def __init__(self, in_channels, out_channels, stride=(1, 2, 2)):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
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
    Lightweight 3D Res-SE CNN (~2.6M parameters) optimized for 24/7 CCTV Crime Detection.
    Maintains exact layer naming and output dimensions for 100% checkpoint compatibility.
    """
    def __init__(self, input_channels=3):
        super(CNN3D_ResSE, self).__init__()

        # Entry Convolutional Stem
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        # Residual Stages with Squeeze-and-Excitation Motion Attention
        self.layer1 = ResBlock3D(32, 32, stride=(1, 1, 1))
        self.layer2 = ResBlock3D(32, 64, stride=(1, 2, 2))
        self.layer3 = ResBlock3D(64, 128, stride=(1, 2, 2))
        self.layer4 = ResBlock3D(128, 256, stride=(2, 2, 2))

        # Global Spatiotemporal Average Pooling & Classification Head
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
# COMPREHENSIVE EVALUATION METRICS ENGINE
# ==============================================================================

def compute_detailed_metrics(y_true, y_probs, threshold=0.5):
    """
    Computes a full suite of binary classification metrics:
    Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrix.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0

    # ROC-AUC and Average Precision
    try:
        roc_auc = roc_auc_score(y_true, y_probs) * 100.0
    except ValueError:
        roc_auc = 0.0

    try:
        pr_auc = average_precision_score(y_true, y_probs) * 100.0
    except ValueError:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }


def find_optimal_threshold(y_true, y_probs):
    """
    Grid searches candidate decision thresholds [0.10, 0.90] to discover the threshold
    that maximizes the validation F1-Score.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    best_thresh = 0.5
    best_f1 = 0.0

    for thresh in np.linspace(0.10, 0.90, 81):
        preds = (y_probs >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)

    return best_thresh, best_f1 * 100.0


# ==============================================================================
# TRAINING & EVALUATION ROUTINES
# ==============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, label_smoothing=Config.LABEL_SMOOTHING):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(train_loader, desc="Training", unit="batch", leave=False)
    for inputs, labels, _ in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Binary label smoothing: 0 -> eps, 1 -> (1 - eps)
        if label_smoothing > 0.0:
            targets = labels * (1.0 - 2.0 * label_smoothing) + label_smoothing
        else:
            targets = labels

        if device.type == "cuda":
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.CLIP_GRAD_NORM)
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(outputs)
        predicted = (probs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

    return running_loss / total, 100.0 * correct / total


def eval_epoch(model, dataloader, criterion, device, threshold=0.5):
    """
    Evaluates model over an entire DataLoader, gathering probabilities and computing
    comprehensive metrics.
    """
    model.eval()
    running_loss = 0.0
    total = 0

    y_true_list = []
    y_probs_list = []
    video_idx_list = []

    with torch.no_grad():
        for inputs, labels, v_indices in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            total += labels.size(0)

            y_true_list.extend(labels.cpu().numpy().flatten())
            y_probs_list.extend(probs.cpu().numpy().flatten())
            video_idx_list.extend(v_indices.numpy().flatten())

    avg_loss = running_loss / max(total, 1)
    y_true = np.array(y_true_list)
    y_probs = np.array(y_probs_list)
    video_indices = np.array(video_idx_list)

    metrics = compute_detailed_metrics(y_true, y_probs, threshold=threshold)
    metrics["loss"] = float(avg_loss)

    return avg_loss, metrics, y_true, y_probs, video_indices


def evaluate_video_level(y_true_clips, y_probs_clips, video_indices, threshold=0.5):
    """
    Aggregates clip-level predictions to whole video verdicts:
    Computes both mean probability and max probability pooling per video.
    """
    unique_videos = np.unique(video_indices)
    video_labels = []
    video_mean_probs = []
    video_max_probs = []

    for v_id in unique_videos:
        mask = (video_indices == v_id)
        # Ground truth label is identical for all clips of the same video
        v_label = y_true_clips[mask][0]
        v_probs = y_probs_clips[mask]

        video_labels.append(v_label)
        video_mean_probs.append(np.mean(v_probs))
        video_max_probs.append(np.max(v_probs))

    video_labels = np.array(video_labels)
    video_mean_probs = np.array(video_mean_probs)
    video_max_probs = np.array(video_max_probs)

    mean_metrics = compute_detailed_metrics(video_labels, video_mean_probs, threshold=threshold)
    max_metrics = compute_detailed_metrics(video_labels, video_max_probs, threshold=threshold)

    return {
        "total_videos": len(unique_videos),
        "mean_pooling": mean_metrics,
        "max_pooling": max_metrics
    }


# ==============================================================================
# MAIN TRAINING & EVALUATION PIPELINE
# ==============================================================================
def main():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 1. Dry Run Flag Check
    if "--dry-run" in sys.argv:
        print("\n--- DRY RUN SANITY CHECK ---")
        model = CNN3D_ResSE().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"3D Res-SE CNN Model Loaded. Total Parameters: {total_params:,}")
        dummy_input = torch.randn(
            2, 3, Config.MAX_FRAMES, Config.TARGET_SIZE[0], Config.TARGET_SIZE[1]
        ).to(device)
        with torch.no_grad():
            out = model(dummy_input)
        print(f"Forward pass output shape: {out.shape} (Expected: [2, 1])")
        print("Dry run completed successfully.")
        return

    # 2. Prepare stratified datasets
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = prepare_dataset()

    # Use stride=12 for training to extract more clips (better coverage), stride=24 for val/test
    train_dataset = FullVideoClipDataset(
        train_paths, train_labels, augment=True, stride=12
    )
    val_dataset = FullVideoClipDataset(
        val_paths, val_labels, augment=False, stride=Config.MAX_FRAMES
    )
    test_dataset = FullVideoClipDataset(
        test_paths, test_labels, augment=False, stride=Config.MAX_FRAMES
    )

    num_workers = min(4, os.cpu_count() or 1) if device.type == "cuda" else 0
    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True  # Prevent single-sample batchnorm crash on last batch
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )

    # 3. Model Initialization
    model = CNN3D_ResSE().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Initialized: 3D Res-SE CNN (Total Parameters: {total_params:,})")

    # Class balance calculation for positive loss weighting
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    pos_weight = torch.tensor([float(num_neg) / max(float(num_pos), 1.0)]).to(device)
    print(f"Dataset Class Balance: {num_neg} Non-Violent vs {num_pos} Violent | Pos Weight: {pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Check for Evaluation-Only Mode
    if "--eval-only" in sys.argv:
        print("\n" + "=" * 65)
        print("EVAL-ONLY MODE: Evaluating existing checkpoint on Test Set")
        print("=" * 65)
        if not os.path.exists(Config.BEST_MODEL_PATH):
            print(f"ERROR: Checkpoint not found at '{Config.BEST_MODEL_PATH}'. Run training first.")
            return

        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device, weights_only=True))
        run_full_evaluation(model, val_loader, test_loader, criterion, device)
        return

    # 5. Optimizer & Learning Rate Scheduler with Warmup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.BASE_LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing with Warmup schedule
    warmup_epochs = Config.WARMUP_EPOCHS
    total_epochs = Config.EPOCHS

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return Config.MIN_LR / Config.BASE_LR + 0.5 * (1.0 - Config.MIN_LR / Config.BASE_LR) * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # 6. Training Loop with Multi-Metric Checkpointing
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_threshold = 0.5
    no_improve_count = 0
    history = []

    print("\nStarting Training Pipeline...")
    print("=" * 95)
    print(f"{'Epoch':^8} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^9} | {'Val Acc':^8} | {'Val F1':^7} | {'Val AUC':^8} | {'LR':^9}")
    print("=" * 95)

    for epoch in range(Config.EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics, val_true, val_probs, _ = eval_epoch(model, val_loader, criterion, device, threshold=0.5)
        scheduler.step()

        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1_score"]
        val_auc = val_metrics["roc_auc"]

        # Track history
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "val_auc": float(val_auc),
            "lr": float(current_lr)
        })

        print(
            f"[{epoch+1:02d}/{Config.EPOCHS:02d}]   | "
            f"{train_loss:^10.4f} | {train_acc:^9.2f}% | "
            f"{val_loss:^9.4f} | {val_acc:^7.2f}% | {val_f1:^7.2f}% | {val_auc:^8.2f}% | "
            f"{current_lr:^9.2e}"
        )

        # Optimize validation decision threshold
        epoch_thresh, epoch_opt_f1 = find_optimal_threshold(val_true, val_probs)

        # Model Checkpointing: Save if Validation F1 improves or composite performance advances
        is_best = (val_f1 > best_val_f1) or (abs(val_f1 - best_val_f1) < 0.5 and val_loss < best_val_loss)

        if is_best and val_f1 > 0:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_threshold = epoch_thresh
            no_improve_count = 0

            # 1. Save standard state_dict checkpoint (100% compatible with test_Model.py)
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

            # 2. Save full checkpoint with optimizer state and metadata
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "optimal_threshold": best_threshold,
                "val_metrics": val_metrics
            }, Config.FULL_CHECKPOINT_PATH)

            print(f"  ★ Best Checkpoint Saved! (Val F1: {best_val_f1:.2f}% | Val Acc: {best_val_acc:.2f}% | Best Thresh: {best_threshold:.2f})")
        else:
            no_improve_count += 1

        # Save training metrics history
        with open(Config.METRICS_LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)

        # ─── Early Stopping Logic ─────────────────────────────────────────────
        # Overfitting check: train acc >> val acc by more than threshold
        overfit_gap = train_acc - val_acc
        if overfit_gap > Config.OVERFIT_GAP_THRESHOLD and epoch >= 20:
            print(
                f"\n[Early Stop] Overfitting gap reached {overfit_gap:.2f}% "
                f"(Train: {train_acc:.2f}% >> Val: {val_acc:.2f}%). Stopping."
            )
            break

        # Patience check: stop if validation F1 hasn't improved for EARLY_STOP_PATIENCE epochs
        if no_improve_count >= Config.EARLY_STOP_PATIENCE:
            print(
                f"\n[Early Stop] Validation performance did not improve for "
                f"{Config.EARLY_STOP_PATIENCE} consecutive epochs (best F1: {best_val_f1:.2f}%). Stopping."
            )
            break

    # 7. Final Comprehensive Evaluation on Test Set
    run_full_evaluation(model, val_loader, test_loader, criterion, device, best_threshold)


def run_full_evaluation(model, val_loader, test_loader, criterion, device, optimal_thresh=0.5):
    """
    Executes a multi-tier evaluation on the holdout test set:
    - Clip-Level Evaluation at default threshold (0.50)
    - Clip-Level Evaluation at optimal threshold
    - Video-Level Aggregated Evaluation (Mean Pooling & Max Pooling)
    - Saves comprehensive report to models1/evaluation_report.json
    """
    print("\n" + "=" * 65)
    print("EVALUATING BEST MODEL CHECKPOINT ON HOLDOUT TEST SET")
    print("=" * 65)

    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=device, weights_only=True)
        )
        print(f"Loaded weights from '{Config.BEST_MODEL_PATH}'.")

    # If optimal threshold was not provided, calibrate it on validation set
    if optimal_thresh == 0.5:
        _, _, val_true, val_probs, _ = eval_epoch(model, val_loader, criterion, device)
        optimal_thresh, _ = find_optimal_threshold(val_true, val_probs)

    # Evaluate on Test Set
    test_loss, test_metrics_default, y_true, y_probs, video_indices = eval_epoch(
        model, test_loader, criterion, device, threshold=0.5
    )
    test_metrics_optimal = compute_detailed_metrics(y_true, y_probs, threshold=optimal_thresh)

    # Video-Level Aggregation
    video_results_default = evaluate_video_level(y_true, y_probs, video_indices, threshold=0.5)
    video_results_optimal = evaluate_video_level(y_true, y_probs, video_indices, threshold=optimal_thresh)

    # --------------------------------------------------------------------------
    # PRINT FORMATTED EVALUATION RESULTS
    # --------------------------------------------------------------------------
    print("\n--- 1. CLIP-LEVEL METRICS (Standard Threshold = 0.50) ---")
    print(f"  Test Loss            : {test_loss:.4f}")
    print(f"  Test Accuracy        : {test_metrics_default['accuracy']:.2f}%")
    print(f"  Precision (Violence) : {test_metrics_default['precision']:.2f}%")
    print(f"  Recall (Violence)    : {test_metrics_default['recall']:.2f}%")
    print(f"  Specificity (Normal) : {test_metrics_default['specificity']:.2f}%")
    print(f"  F1-Score             : {test_metrics_default['f1_score']:.2f}%")
    print(f"  ROC-AUC Score        : {test_metrics_default['roc_auc']:.2f}%")
    print(f"  PR-AUC (Avg Prec)    : {test_metrics_default['pr_auc']:.2f}%")
    cm = test_metrics_default['confusion_matrix']
    print(f"  Confusion Matrix     : TN={cm['TN']} | FP={cm['FP']} | FN={cm['FN']} | TP={cm['TP']}")

    print(f"\n--- 2. CLIP-LEVEL METRICS (Calibrated Optimal Threshold = {optimal_thresh:.2f}) ---")
    print(f"  Calibrated Accuracy  : {test_metrics_optimal['accuracy']:.2f}%")
    print(f"  Calibrated Precision : {test_metrics_optimal['precision']:.2f}%")
    print(f"  Calibrated Recall    : {test_metrics_optimal['recall']:.2f}%")
    print(f"  Calibrated F1-Score  : {test_metrics_optimal['f1_score']:.2f}%")
    cm_opt = test_metrics_optimal['confusion_matrix']
    print(f"  Confusion Matrix     : TN={cm_opt['TN']} | FP={cm_opt['FP']} | FN={cm_opt['FN']} | TP={cm_opt['TP']}")

    print("\n--- 3. WHOLE VIDEO-LEVEL VERDICT (Aggregated Surveillance Performance) ---")
    print(f"  Total Test Videos    : {video_results_default['total_videos']}")
    v_mean = video_results_default['mean_pooling']
    print(f"  Video Accuracy (Mean): {v_mean['accuracy']:.2f}% | F1: {v_mean['f1_score']:.2f}% | Recall: {v_mean['recall']:.2f}%")
    v_max = video_results_default['max_pooling']
    print(f"  Video Accuracy (Max) : {v_max['accuracy']:.2f}% | F1: {v_max['f1_score']:.2f}% | Recall: {v_max['recall']:.2f}%")

    print("\n--- 4. DETAILED CLASSIFICATION REPORT (Default Threshold 0.5) ---")
    y_pred = (y_probs >= 0.5).astype(int)
    print(classification_report(y_true, y_pred, target_names=['Non-Violent', 'Violent'], digits=4))

    # Save full evaluation report to JSON
    report_data = {
        "optimal_threshold": float(optimal_thresh),
        "clip_level_default": test_metrics_default,
        "clip_level_optimal": test_metrics_optimal,
        "video_level_default": video_results_default,
        "video_level_optimal": video_results_optimal
    }

    with open(Config.EVAL_REPORT_PATH, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[Saved] Full evaluation report written to '{Config.EVAL_REPORT_PATH}'.")


if __name__ == "__main__":
    main()
