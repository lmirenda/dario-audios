import whisper

model = whisper.load_model("base")
result = model.transcribe("audio_1.m4a")
print(result["text"])
