import time

import whisper


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' {func.__params__} took {elapsed_time:.6f} seconds to execute.")
        return result

    return wrapper


@time_it
def run(load_model: str):
    model = whisper.load_model(load_model)
    result = model.transcribe("audios/test_1.ogg", fp16=False)
    print(result["text"])


if __name__ == '__main__':
    run("tiny")
    run("base")
    run("medium")
    run("large")
