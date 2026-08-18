from faster_whisper import WhisperModel

# =====================================
# 配置
# =====================================

AUDIO_FILE = r"D:\code_center\video_process\FCK_Final.mp4"

OUTPUT_SRT = r"D:\code_center\video_process\fck_en_new.srt"

MODEL_PATH = r"D:\code_center\models\faster-whisper-large-v3"

# =====================================
# SRT时间格式
# =====================================

def format_time(seconds):

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    ms = int(round((seconds - int(seconds)) * 1000))

    if ms == 1000:
        ms = 0
        s += 1

    return f"{h:02}:{m:02}:{s:02},{ms:03}"


# =====================================
# 加载模型
# =====================================

print("正在加载模型...")

model = WhisperModel(
    MODEL_PATH,
    device="cuda",
    compute_type="float16"
)

print("模型加载完成")


# =====================================
# 转录
# =====================================

print("开始识别音频...")

segments, info = model.transcribe(
    AUDIO_FILE,
    language="en",
    beam_size=10,
    vad_filter=False
)

# 转成列表
segments = list(segments)

print(f"检测语言: {info.language}")
print(f"语言置信度: {info.language_probability:.2f}")
print(f"识别片段数: {len(segments)}")


# =====================================
# 输出SRT
# =====================================

print("生成字幕文件...")

with open(OUTPUT_SRT, "w", encoding="utf-8") as f:

    for idx, seg in enumerate(segments, start=1):

        text = seg.text.strip()

        if not text:
            continue

        f.write(f"{idx}\n")
        f.write(
            f"{format_time(seg.start)} --> "
            f"{format_time(seg.end)}\n"
        )
        f.write(f"{text}\n\n")

print("\n========================")
print("✅ 字幕生成完成")
print(f"输出文件: {OUTPUT_SRT}")
print(f"字幕条数: {len(segments)}")
print("========================")