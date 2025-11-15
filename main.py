from SubtitleGenerator import SubtitleGenerator
import argparse

def main():
    parser = argparse.ArgumentParser(description="Video Subtitle Generator (Offline)")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--model_name", default="turbo",
                        choices=["tiny", "base", "small", "medium", "large", "turbo"],
                        help="Whisper model name")
    parser.add_argument("--translate", choices=["fr", "en"], help="Translate subtitles to (fr or en)")

    args = parser.parse_args()

    print(f"Processing video: {args.video_path}")
    print(f"Whisper model: {args.model_name}")
    if args.translate:
        print(f"Translating subtitles to: {args.translate}")

    # Ensure your NLLB local path matches the folder where you saved the model
    nllb_local_path = r"D:\AI-W-Subtitles\models\nllb-200-distilled-600M"

    generator = SubtitleGenerator(args.model_name, nllb_local_path)
    generator.process_video(args.video_path, args.translate)

if __name__ == "__main__":
    main()
