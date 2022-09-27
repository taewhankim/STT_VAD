import os
from copy import deepcopy
import torch
import onnxruntime
from utils_vad import init_jit_model, get_speech_ts_adaptive, get_language, read_audio, _read_other_format
from pathlib import Path
import warnings
import wave
from define import *


warnings.simplefilter("ignore", UserWarning)


def language_classifier(wav):

    def init_onnx_model(model_path: str):
        return onnxruntime.InferenceSession(model_path)

    def validate_onnx(model, inputs):
        with torch.no_grad():
            ort_inputs = {'input': inputs.cpu().numpy()}
            outs = model.run(None, ort_inputs)
            outs = [torch.Tensor(x) for x in outs]
        return outs

    model = init_onnx_model('files/number_detector.onnx')
    language = get_language(wav, model, run_function=validate_onnx)
    return language


def frame2second(stamps, sr=16000):
    for s in stamps:
        s['start'] = s['start']/sr
        s['end'] = s['end']/sr
    return stamps


def save_text(stamps, out_fn):
    with open(out_fn, 'w') as f:
        for line in stamps:
            f.write(f"{str(line['start'])}-{str(line['end'])}\n")


def refine_timestamps(args, timestamps):
    refined_timestamps = []

    for idx, t in enumerate(timestamps):
        if refined_timestamps:
            pre_time = refined_timestamps.pop()
            if args.sample_rate * args.speech_interval > t['start'] - pre_time['end']:
                refined_timestamps.append({'start': pre_time['start'], 'end': t['end']})
            else:
                refined_timestamps.append(pre_time)
                refined_timestamps.append(t)
        else:
            refined_timestamps.append(t)
            continue

    return refined_timestamps

def get_sample_rate(path: str):
    with wave.open(path, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
    return frame_rate

def main_vad(args, **kwargs):
    logger = kwargs.get("logger")
    if Path(args.input_fn).suffix != '.wav':
        logger.debug(f"input file: {args.input_fn} format is not wav")
        args.input_fn = _read_other_format(args.input_fn)

    logger.debug(f"default sample_rate: {args.sample_rate}")
    # TODO: 차후 수정
    # args.sample_rate = get_sample_rate(args.input_fn)
    logger.debug(f"now file's sample_rate: {args.sample_rate}")

    wav = read_audio(args.input_fn, target_sr=args.sample_rate)
    model = init_jit_model(model_path=args.vad_model)

    frame_stamps = get_speech_ts_adaptive(wav, model,
                                          num_samples_per_window=int(args.sample_rate/4),
                                          min_speech_samples=int(args.sample_rate/2),
                                          min_silence_samples=int(args.sample_rate/4),
                                          speech_pad_samples=int(args.sample_rate/8))
    frame_stamps = refine_timestamps(args, frame_stamps)

    time_stamps = frame2second(deepcopy(frame_stamps))

    out_fn = os.path.join(args.output_dir, f'{Path(args.input_fn).stem}_speech_timestamps.txt')
    save_text(time_stamps, out_fn)

    logger.debug(f"split time stamps where someone talks in wav file")
    return frame_stamps, wav


def tuning_speech_parameters(parameters: dict, now_sample_rate: int):
    constant = now_sample_rate / parameters.get("sample_rate")
    parameters["constant"] = constant
    parameters["sample_rate"] = now_sample_rate
    parameters["batch_size"] = int(parameters["batch_size"] * constant)
    parameters["step"] = int(parameters["step"] * constant)
    parameters["speech_interval"] = float(f"{(parameters['speech_interval'] * constant):.3f}")
    parameters["num_samples_per_window"] = int(parameters["num_samples_per_window"] * constant)
    parameters["min_speech_samples"] = int(parameters["min_speech_samples"] * constant)
    parameters["min_silence_samples"] = int(parameters["min_silence_samples"] * constant)
    parameters["speech_pad_samples"] = int(parameters["speech_pad_samples"] * constant)

    return parameters



# 주희팀장님이 만드신 함수에서, text save 하는 부분만 제외
def get_time_stamps(args, **kwargs):
    logger = kwargs.get("logger")
    device = kwargs.get("device")

    if Path(args.input_fn).suffix != '.wav':
        logger.debug(f"input file: {args.input_fn} format is not wav")
        args.input_fn = _read_other_format(args.input_fn)

    wav = read_audio(args.input_fn, target_sr=args.sample_rate)

    model = init_jit_model(model_path=args.vad_model, device=device)
    # 2021.05.04 주희팀장님 피드백: 원 sample rate 유지
    # parameters = tuning_speech_parameters(kwargs.get("default_adaptive_parameters"), args.sample_rate)

    parameters = default_adaptive_parameters
    # logger.debug(f"tuning parameters: {parameters}")
    if args.lang=='ko':
        parameters['speech_interval'] = args.speech_interval
    logger.debug(f"default parameters: {default_adaptive_parameters}")

    # adaptive 에 들어가는 파라미터들은 default 값 유지, 차후 변경점이 있으면, tuning_speech_parameters 주석 해제후 진행
    frame_stamps = get_speech_ts_adaptive(wav, model,
                                          step = parameters.get("step"),
                                          num_samples_per_window=parameters.get("num_samples_per_window"),
                                          min_speech_samples=parameters.get("min_speech_samples"),
                                          min_silence_samples=parameters.get("min_silence_samples"),
                                          speech_pad_samples=parameters.get("speech_pad_samples"),
                                          device=device)

    # frame_stamps = refine_timestamps(args, frame_stamps)

    logger.debug(f"split time stamps where someone talks in wav file")
    return frame_stamps