from faster_whisper import WhisperModel
import asyncio
from googletrans import Translator
# Step 1: Load local faster-whisper model
model_path = r"D:\code_center\hugface"  # Use raw string for Windows paths
model = WhisperModel(model_path, device="cpu")  # Change to "cuda" if you have GPU

# Transcribe English audio
segments, info = model.transcribe("audio.wav", language="en")
print(f"Detected language: {info.language}, Duration: {info.duration:.2f}s")

translator = Translator()

# Helper: format time for SRT
def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02}:{mins:02}:{secs:02},{millis:03}"

async def main():
    tasks = []

    # Prepare translation tasks
    for i, seg in enumerate(segments, start=1):
        start = format_time(seg.start)
        end = format_time(seg.end)
        text = seg.text.strip()
        tasks.append((i, start, end, text))

    # Translate asynchronously
    async def translate_segment(i, start, end, text):
        translated = await translator.translate(text, src="en", dest="zh-cn")
        return f"{i}\n{start} --> {end}\n{translated.text}\n\n"

    srt_results = await asyncio.gather(*[translate_segment(*t) for t in tasks])

    # Write SRT file
    with open("subtitles.srt", "w", encoding="utf-8") as f:
        f.writelines(srt_results)

    print("✅ SRT file generated: subtitles.srt")

# Run async translation
asyncio.run(main())
