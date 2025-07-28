import re
import time
from deep_translator import GoogleTranslator
from langdetect import detect

def translate_srt(input_file, output_file):
    # Read the input SRT file
    print(f"Reading file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("❌ Error: Input SRT file not found!")
        return
    except Exception as e:
        print(f"❌ Error reading the file: {e}")
        return

    print("✅ File read successfully. Starting translation...\n")

    translated_lines = []
    for line in lines:
        if re.match(r'^\d+$', line.strip()) or '-->' in line:  # Keep timestamps
            translated_lines.append(line)
        elif line.strip():  # Check non-empty subtitle text
            try:
                detected_lang = detect(line.strip())  # Detect language
                if detected_lang == 'en':  # If already English, keep it
                    translated_lines.append(line)
                    print(f"✅ Skipped (English): {line.strip()}\n")
                else:  # Translate non-English text
                    translator = GoogleTranslator(source=detected_lang, target='en')
                    translated_text = translator.translate(line.strip())
                    translated_lines.append(translated_text + '\n')
                    print(f"🔄 Translated ({detected_lang} → en): {translated_text}\n")
                    time.sleep(1)  # Prevent API limit issues
            except Exception as e:
                print(f"❌ Language detection/translation error: {e}")
                translated_lines.append(line)  # Keep original line in case of error
        else:
            translated_lines.append('\n')

    # Save the translated subtitles
    print(f"\n💾 Saving translated subtitles to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.writelines(translated_lines)
        print("🎉 Translation completed successfully!")
    except Exception as e:
        print(f"❌ Error writing to file: {e}")

# File paths
input_srt = r""
output_srt = r""

translate_srt(input_srt, output_srt)
