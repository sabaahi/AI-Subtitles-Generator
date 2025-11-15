import whisper
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
from datetime import timedelta
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
import sys

class SubtitleGenerator:
    def __init__(self, whisper_model_name: str = "large-v3",
                 nllb_local_path: str = r"D:\AI-W-Subtitles\models\nllb-200-distilled-600M"):
        """Initialize Whisper and NLLB models from local paths for offline use."""

        # Load Whisper
        print(f"Loading Whisper model '{whisper_model_name}'...")
        self.whisper_model = whisper.load_model(whisper_model_name)

        # Load NLLB for offline translation
        if Path(nllb_local_path).exists():
            print(f"Loading NLLB model from '{nllb_local_path}'...")
            self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_local_path, local_files_only=True)
            self.nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_local_path, local_files_only=True)
        else:
            print(f"NLLB model not found at '{nllb_local_path}'. Translation will be disabled.")
            self.nllb_model = None
            self.nllb_tokenizer = None

    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, codec="pcm_s16le")
            video.close()
            return True
        except Exception as e:
            print(f"Error during audio extraction: {str(e)}")
            return False

    def generate_subtitles(self, audio_path: str) -> tuple:
        try:
            print("Transcribing audio with Whisper...")
            result = self.whisper_model.transcribe(audio_path, verbose=False)
            source_lang = result["language"]
            print(f"Detected language: {source_lang}")
            return result, source_lang
        except Exception as e:
            print(f"Error during subtitle generation: {str(e)}")
            return None, None

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if self.nllb_model is None or self.nllb_tokenizer is None:
            return text
        try:
            inputs = self.nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = self.nllb_model.generate(**inputs)
            return self.nllb_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during translation: {str(e)}")
            return text

    @staticmethod
    def format_time(seconds: float) -> str:
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds // 60) % 60
        seconds = td.seconds % 60
        milliseconds = round(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def create_srt(self, transcription: dict, output_path: str, target_lang: str = None) -> bool:
        try:
            source_lang = transcription["language"]
            with open(output_path, "w", encoding="utf-8") as f:
                segments = transcription["segments"]
                for i, segment in tqdm(enumerate(segments, 1), desc="Creating SRT file"):
                    start_time = self.format_time(segment["start"])
                    end_time = self.format_time(segment["end"])
                    text = segment["text"].strip()
                    if target_lang and source_lang != target_lang:
                        text = self.translate_text(text, source_lang, target_lang)
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
            return True
        except Exception as e:
            print(f"Error during SRT file creation: {str(e)}")
            return False

    def process_video(self, video_path: str, target_lang: str = None) -> None:
        output_dir = "subtitles"
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, "temp_audio.wav")

        print("Starting subtitle generation process...")
        print("Extracting audio...")
        if not self.extract_audio(video_path, audio_path):
            return

        print("Generating subtitles...")
        transcription, source_lang = self.generate_subtitles(audio_path)
        if transcription is None:
            return

        base_name = os.path.splitext(os.path.basename(video_path))[0]

        original_srt = os.path.join(output_dir, f"{base_name}_{source_lang}.srt")
        self.create_srt(transcription, original_srt)
        print(f"Original subtitles generated: {original_srt}")

        if target_lang:
            translated_srt = os.path.join(output_dir, f"{base_name}_{target_lang}.srt")
            self.create_srt(transcription, translated_srt, target_lang)
            print(f"Translated subtitles generated: {translated_srt}")

        if os.path.exists(audio_path):
            os.remove(audio_path)

        print("Process completed!")
