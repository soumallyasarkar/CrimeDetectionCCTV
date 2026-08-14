import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "Dataset"

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".wmv",
    ".mpg",
    ".mpeg"
)

MAX_WORKERS = 4

# ============================================================
# FIND ALL VIDEOS
# ============================================================

def find_videos():

    videos = []

    for root, dirs, files in os.walk(DATASET_DIR):

        for filename in files:

            if filename.lower().endswith(VIDEO_EXTENSIONS):

                videos.append(
                    os.path.join(root, filename)
                )

    return sorted(videos)


# ============================================================
# CHECK ONE VIDEO USING FFMPEG
# ============================================================

def check_video(video_path):

    try:

        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                video_path,
                "-f",
                "null",
                "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        error_output = result.stderr.strip()

        if error_output:

            return {
                "path": video_path,
                "bad": True,
                "error": error_output
            }

        return {
            "path": video_path,
            "bad": False,
            "error": ""
        }

    except FileNotFoundError:

        return {
            "path": video_path,
            "bad": True,
            "error": "FFmpeg is not installed or not found in PATH."
        }

    except Exception as e:

        return {
            "path": video_path,
            "bad": True,
            "error": str(e)
        }


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    print("=" * 70)
    print("              VIDEO DATASET CHECKER")
    print("=" * 70)

    print("\nDataset directory:", DATASET_DIR)

    videos = find_videos()

    print("Total videos found:", len(videos))

    if len(videos) == 0:

        print("\nNo videos found.")
        return

    print("\nScanning videos using FFmpeg...")
    print("-" * 70)

    bad_videos = []

    completed = 0
    total = len(videos)

    # --------------------------------------------------------
    # Parallel scanning
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                check_video,
                video
            ): video
            for video in videos
        }

        for future in as_completed(futures):

            result = future.result()

            completed += 1

            print(
                f"\rProgress: "
                f"{completed}/{total}",
                end="",
                flush=True
            )

            if result["bad"]:

                bad_videos.append(result)

    print("\n")

    # ========================================================
    # RESULTS
    # ========================================================

    print("=" * 70)
    print("                 SCAN COMPLETE")
    print("=" * 70)

    print(
        f"Total videos : {total}"
    )

    print(
        f"Good videos  : {total - len(bad_videos)}"
    )

    print(
        f"Bad videos   : {len(bad_videos)}"
    )

    print("=" * 70)

    # ========================================================
    # BAD VIDEOS
    # ========================================================

    if len(bad_videos) == 0:

        print("\nNo FFmpeg decoding errors detected.")

        print(
            "\nThis means the H.264 message may be a "
            "recoverable decoder warning."
        )

        return

    print("\nPROBLEMATIC VIDEOS:")
    print("=" * 70)

    for i, result in enumerate(
        bad_videos,
        start=1
    ):

        print(f"\n[{i}] {result['path']}")

        print("Error:")

        print(result["error"])

    # ========================================================
    # SAVE BAD VIDEO PATHS
    # ========================================================

    with open(
        "bad_videos.txt",
        "w",
        encoding="utf-8"
    ) as f:

        for result in bad_videos:

            f.write(
                result["path"] + "\n"
            )

    # ========================================================
    # SAVE DETAILED REPORT
    # ========================================================

    with open(
        "video_errors.txt",
        "w",
        encoding="utf-8"
    ) as f:

        for result in bad_videos:

            f.write("=" * 70 + "\n")

            f.write(
                result["path"] + "\n"
            )

            f.write("=" * 70 + "\n")

            f.write(
                result["error"] + "\n\n"
            )

    print("\n" + "=" * 70)

    print(
        "Bad video paths saved to : bad_videos.txt"
    )

    print(
        "Detailed errors saved to : video_errors.txt"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()