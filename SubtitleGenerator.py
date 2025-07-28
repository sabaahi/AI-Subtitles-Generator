import whisper
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
from datetime import timedelta
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class SubtitleGenerator:
    def __init__(self, model_name: str = "turbo"):
        """Initialize the SubtitleGenerator with Whisper model and translation models.

        Args:
            model_name (str): The model size to use for Whisper. Default is "turbo".
        """
        self.whisper_model = whisper.load_model("large-v3")
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        self.nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from a video file.

        Args:
            video_path (str): Path to the input video file.
            audio_path (str): Path to save the extracted audio.

        Returns:
            bool: True if audio extraction was successful, False otherwise.
        """
        try:
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, codec="pcm_s16le")
            video.close()
            return True
        except Exception as e:
            print(f"Error during audio extraction: {str(e)}")
            return False

    def detect_language(self, audio_path: str) -> str:
        """Detect the language of the given audio using Whisper.

        Args:
            audio_path (str): Path to the extracted audio file.

        Returns:
            str: The detected language code.
        """
        try:
            result = self.whisper_model.transcribe(audio_path, verbose=False)
            return result["language"]
        except Exception as e:
            print(f"Error detecting language: {str(e)}")
            return None

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate the given text from source language to target language.

        Args:
            text (str): The text to translate.
            source_lang (str): The source language code (e.g., 'en' or 'fr').
            target_lang (str): The target language code (e.g., 'en' or 'fr').

        Returns:
            str: The translated text.
        """
        try:
            inputs = self.nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = self.nllb_model.generate(**inputs)
            return self.nllb_tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error during translation: {str(e)}")
            return text

    def generate_subtitles(self, audio_path: str) -> tuple:
        """Generate subtitles from the audio file using Whisper.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            tuple: A tuple containing the transcription result and the detected language.
        """
        try:
            result = self.whisper_model.transcribe(audio_path, verbose=False)
            source_lang = result["language"]
            print(f"Detected language: {source_lang}")
            return result, source_lang
        except Exception as e:
            print(f"Error during subtitle generation: {str(e)}")
            return None, None

    @staticmethod
    def format_time(seconds: float) -> str:
        """Convert time in seconds to SRT format.

        Args:
            seconds (float): The time in seconds to convert.

        Returns:
            str: The time in SRT format (hh:mm:ss,ms).
        """
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds // 60) % 60
        seconds = td.seconds % 60
        milliseconds = round(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def create_srt(self, transcription: dict, output_path: str, target_lang: str = None) -> bool:
        """Create an SRT file with optional translation.

        Args:
            transcription (dict): The transcription result from Whisper.
            output_path (str): The path where the SRT file will be saved.
            target_lang (str, optional): The target language for translation. Defaults to None.

        Returns:
            bool: True if SRT file creation was successful, False otherwise.
        """
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
        """Main process to generate subtitles from the video.

        Args:
            video_path (str): Path to the video file.
            target_lang (str, optional): The language to translate subtitles to. Defaults to None.
        """
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
