
import wave
from define import *
import soundfile as sf


def limit_numbers(numbers: float, limit=4):
    temp = f"{numbers:.{limit}f}"
    return float(temp)


def get_times_of_file(path: str) -> float:
    file = sf.SoundFile(path)
    seconds = len(file) / file.samplerate
    return seconds


def post_processing(args,total_text) -> list:

    results = []
    for sample_index, sample in enumerate(args.timestamps):
        refine = {
            "start": sample.get("start") / args.sample_rate,
            "end": sample.get("end") / args.sample_rate,
            "text": []
        }
        # words = []

        # for word_index, word_info in enumerate(sample.get("text")):
        #     words.append(word_info.get("word"))
        # refine['text'] = " ".join(words)
        ## 수정
        refine['text'] = total_text[sample_index]
        # refine['text'] = sample['text']

        results.append(refine)

    return results

def get_sample_rate(path: str) -> int:
    with wave.open(path, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
    return frame_rate

def init_args(args, **kwargs):

    args.lang = kwargs.get("language")
    args.input_fn = kwargs.get("path")
    # 2021.05.04 주희팀장님 피드백: 원 sample rate 16000 고정
    # args.sample_rate = get_sample_rate(args.input_fn)
    args.vad_model = vad_model

    return args

def wrap_data(request_data, text):

    outputs = {
        "Object": [],
        # "path": request_data.get("path"),
        "result": ""
    }

    message = "success" if len(text) != 0 else "error"

    if message == "success":
        outputs = {
            "Object": text,
            # "path": request_data.get("path"),
            "result": message
        }
        return outputs

    elif message == "error":
        outputs["result"] = message
        return outputs