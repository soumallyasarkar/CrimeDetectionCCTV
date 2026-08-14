import cv2
import torch
import numpy as np

from Model import CNN3D_ResSE, Config


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = CNN3D_ResSE().to(device)

model.load_state_dict(
    torch.load(
        Config.BEST_MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# READ ENTIRE VIDEO
# ============================================================

def read_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\nVideo Information")
    print("-" * 40)
    print("FPS          :", fps)
    print("Total Frames :", total_frames)

    if fps > 0:
        print("Duration     :", round(total_frames / fps, 2), "seconds")

    print("-" * 40)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(
            frame,
            Config.TARGET_SIZE
        )

        frame = frame.astype(np.float32) / 255.0

        frames.append(frame)

    cap.release()

    return np.array(frames)


# ============================================================
# MOTION PREPROCESSING
# SAME AS TRAINING
# ============================================================

def compute_motion(frames):

    if len(frames) < 2:
        return frames

    motion = np.abs(
        np.diff(frames, axis=0)
    )

    # Pad first frame
    first_diff = motion[0:1]

    motion = np.concatenate(
        [first_diff, motion],
        axis=0
    )

    # Same preprocessing used during training
    blended = (
        0.7 * frames +
        0.3 * motion
    )

    return blended


# ============================================================
# CREATE 24-FRAME CLIPS
# ============================================================

def create_clips(frames):

    max_frames = Config.MAX_FRAMES

    clips = []

    total_frames = len(frames)

    # Non-overlapping clips
    for start in range(0, total_frames, max_frames):

        end = start + max_frames

        clip = frames[start:end]

        # If final clip is shorter than 24 frames,
        # pad it using the last frame
        if len(clip) > 0 and len(clip) < max_frames:

            while len(clip) < max_frames:
                clip = np.concatenate(
                    [clip, clip[-1:]],
                    axis=0
                )

        if len(clip) == max_frames:
            clips.append(clip)

    return clips


# ============================================================
# PREDICT ONE CLIP
# ============================================================

def predict_clip(clip):

    # Calculate motion
    clip = compute_motion(clip)

    # Convert:
    #
    # (T, H, W, C)
    #
    # to:
    #
    # (C, T, H, W)

    tensor = torch.FloatTensor(
        clip
    ).permute(3, 0, 1, 2)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(output).item()

    return probability


# ============================================================
# ANALYZE ENTIRE VIDEO
# ============================================================

def analyze_video(video_path):

    print("\nLoading entire video...")

    frames = read_video(video_path)

    if len(frames) == 0:
        print("ERROR: Could not read video.")
        return

    print("\nTotal frames loaded:", len(frames))

    # Create clips
    clips = create_clips(frames)

    print("Total clips:", len(clips))
    print("\nAnalyzing video...")
    print("=" * 50)

    probabilities = []

    for i, clip in enumerate(clips):

        probability = predict_clip(clip)

        probabilities.append(probability)

        prediction = "VIOLENT" if probability >= 0.5 else "NON-VIOLENT"

        print(
            f"Clip {i + 1:03d}/{len(clips):03d} | "
            f"Violence Probability: {probability * 100:6.2f}% | "
            f"{prediction}"
        )

    # ========================================================
    # FINAL VIDEO DECISION
    # ========================================================

    probabilities = np.array(probabilities)

    average_probability = np.mean(probabilities)

    violent_clips = np.sum(probabilities >= 0.5)

    total_clips = len(probabilities)

    violent_percentage = (
        violent_clips / total_clips
    ) * 100

    # Final decision
    if average_probability >= 0.5:
        final_prediction = "VIOLENT"
    else:
        final_prediction = "NON-VIOLENT"

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("              FINAL VIDEO VERDICT")
    print("=" * 60)

    print(
        f"Total Clips              : {total_clips}"
    )

    print(
        f"Violent Clips            : "
        f"{violent_clips}/{total_clips}"
    )

    print(
        f"Violent Clip Percentage  : "
        f"{violent_percentage:.2f}%"
    )

    print(
        f"Average Violence Score   : "
        f"{average_probability * 100:.2f}%"
    )

    print("-" * 60)

    print(
        f"FINAL PREDICTION         : "
        f"{final_prediction}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    video_path = input(
        "\nEnter video path: "
    ).strip().strip('"')

    try:

        analyze_video(video_path)

    except Exception as e:

        print("\nERROR:", e)