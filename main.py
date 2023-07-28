import whisper

if __name__ == '__main__':

    print("starting")
    model = whisper.load_model("medium")
    result = model.transcribe("audio_1.m4a", fp16=False)
    print(result["text"])
