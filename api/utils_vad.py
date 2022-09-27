import os
import torch
import torchaudio
from collections import deque
import torch.nn.functional as F
from pydub import AudioSegment
from pathlib import Path
from define import stt_languages as languages

import warnings
warnings.simplefilter("ignore", UserWarning)


torchaudio.set_audio_backend("soundfile")  # switch backend


class IterativeMedianMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.median = 0
        self.counts = {}
        for i in range(0, 101, 1):
            self.counts[i / 100] = 0
        self.total_values = 0

    def __call__(self, val):
        self.total_values += 1
        rounded = round(abs(val), 2)
        self.counts[rounded] += 1
        bin_sum = 0
        for j in self.counts:
            bin_sum += self.counts[j]
            if bin_sum >= self.total_values / 2:
                self.median = j
                break
        return self.median


def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

def save_audio(path: str,
               tensor: torch.Tensor,
               sr: int = 16000):
    torchaudio.save(path, tensor.unsqueeze(0), sr)

def read_audio(path: str, target_sr: int = 16000):

    assert torchaudio.get_audio_backend() == 'soundfile'
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = transform(wav)
        sr = target_sr

    assert sr == target_sr
    return wav.squeeze(0)


def init_jit_model(model_path: str,
                   device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def get_speech_ts_adaptive(wav: torch.Tensor,
                      model,
                      batch_size: int = 200,
                      step: int = 500,
                      num_samples_per_window: int = 4000, # Number of samples per audio chunk to feed to NN (4000 for 16k SR, 2000 for 8k SR is optimal)
                      min_speech_samples: int = 10000,  # samples
                      min_silence_samples: int = 4000,
                      speech_pad_samples: int = 2000,
                      run_function=validate,
                      visualize_probs=False,
                      device='cpu'):
    """
    This function is used for splitting long audios into speech chunks using silero VAD
    Attention! All default sample rate values are optimal for 16000 sample rate model, if you are using 8000 sample rate model optimal values are half as much!

    Parameters
    ----------
    batch_size: int
        batch size to feed to silero VAD (default - 200)

    step: int
        step size in samples, (default - 500)

    num_samples_per_window: int
        window size in samples (chunk length in samples to feed to NN, default - 4000)

    min_speech_samples: int
        if speech duration is shorter than this value, do not consider it speech (default - 10000)

    min_silence_samples: int
        number of samples to wait before considering as the end of speech (default - 4000)

    speech_pad_samples: int
        widen speech by this amount of samples each side (default - 2000)

    run_function: function
        function to use for the model call

    visualize_probs: bool
        whether draw prob hist or not (default: False)

    device: string
        torch device to use for the model call (default - "cpu")

    Returns
    ----------
    speeches: list
        list containing ends and beginnings of speech chunks (in samples)
    """

    num_samples = num_samples_per_window
    num_steps = int(num_samples / step)
    assert min_silence_samples >= step
    outs = []
    to_concat = []
    for i in range(0, len(wav), step):
        chunk = wav[i: i+num_samples]
        if len(chunk) < num_samples:
            chunk = F.pad(chunk, (0, num_samples - len(chunk)))
        to_concat.append(chunk.unsqueeze(0))
        if len(to_concat) >= batch_size:
            chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
            out = run_function(model, chunks)
            outs.append(out)
            to_concat = []

    if to_concat:
        chunks = torch.Tensor(torch.cat(to_concat, dim=0)).to(device)
        out = run_function(model, chunks)
        outs.append(out)

    outs = torch.cat(outs, dim=0).cpu()

    buffer = deque(maxlen=num_steps)
    triggered = False
    speeches = []
    smoothed_probs = []
    current_speech = {}
    speech_probs = outs[:, 1]  # 0 index for silence probs, 1 index for speech probs
    median_probs = speech_probs.median()

    trig_sum = 0.89 * median_probs + 0.08 # 0.08 when median is zero, 0.97 when median is 1

    temp_end = 0
    for i, predict in enumerate(speech_probs):
        buffer.append(predict)
        smoothed_prob = max(buffer)
        if visualize_probs:
            smoothed_probs.append(float(smoothed_prob))
        if (smoothed_prob >= trig_sum) and temp_end:
            temp_end = 0
        if (smoothed_prob >= trig_sum) and not triggered:
            triggered = True
            current_speech['start'] = step * max(0, i-num_steps)
            continue
        if (smoothed_prob < trig_sum) and triggered:
            if not temp_end:
                temp_end = step * i
            if step * i - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                temp_end = 0
                current_speech = {}
                triggered = False
                continue
    if current_speech:
        current_speech['end'] = len(wav)
        speeches.append(current_speech)
    # if visualize_probs:
    #     pd.DataFrame({'probs': smoothed_probs}).plot(figsize=(16, 8))

    for ts in speeches:
        ts['start'] = max(0, ts['start'] - speech_pad_samples)
        ts['end'] += speech_pad_samples

    return speeches


def get_number_ts(wav: torch.Tensor,
                  model,
                  model_stride=8,
                  hop_length=160,
                  sample_rate=16000,
                  run_function=validate):
    wav = torch.unsqueeze(wav, dim=0)
    perframe_logits = run_function(model, wav)[0]
    perframe_preds = torch.argmax(torch.softmax(perframe_logits, dim=1), dim=1).squeeze()   # (1, num_frames_strided)
    extended_preds = []
    for i in perframe_preds:
        extended_preds.extend([i.item()] * model_stride)
    # len(extended_preds) is *num_frames_real*; for each frame of audio we know if it has a number in it.
    triggered = False
    timings = []
    cur_timing = {}
    for i, pred in enumerate(extended_preds):
        if pred == 1:
            if not triggered:
                cur_timing['start'] = int((i * hop_length) / (sample_rate / 1000))
                triggered = True
        elif pred == 0:
            if triggered:
                cur_timing['end'] = int((i * hop_length) / (sample_rate / 1000))
                timings.append(cur_timing)
                cur_timing = {}
                triggered = False
    if cur_timing:
        cur_timing['end'] = int(len(wav) / (sample_rate / 1000))
        timings.append(cur_timing)
    return timings


def get_language(wav: torch.Tensor,
                 model,
                 run_function=validate):
    wav = torch.unsqueeze(wav, dim=0)
    lang_logits = run_function(model, wav)[2]
    lang_pred = torch.argmax(torch.softmax(lang_logits, dim=1), dim=1).item()   # from 0 to len(languages) - 1
    assert lang_pred < len(languages)
    return languages[lang_pred]

def collect_chunks(tss: dict,
                   wav: torch.Tensor):
    chunks = []
    for i in tss:
        chunks.append(wav[i['start']: i['end']])
    return torch.cat(chunks)


def _read_other_format(file_name):

    all_video_extension = ['.mp4', '.webm', '.mkv', '.flv', '.avi', '.wmv', '.mpg', '.mpeg']
    all_audio_extension = ['.mp3']
    file_ext = Path(file_name).suffix
    audio_ext = 'wav'
    audio_file_name = file_name.replace(Path(file_name).suffix, '.' + audio_ext)

    if file_ext in all_video_extension:
        EXTRACT_VIDEO_COMMAND = ('ffmpeg -i "{from_video_path}" ' 
                                 '-f {audio_ext} -ab 22050 ' 
                                 '-vn "{to_audio_path}"')
        command = EXTRACT_VIDEO_COMMAND.format(
            from_video_path=file_name, audio_ext=audio_ext, to_audio_path=audio_file_name,
        )
        os.system(command)
    elif file_ext in all_audio_extension:
        if file_ext == '.mp3':
            sound = AudioSegment.from_mp3(file_name)
            sound.export(audio_file_name, format=audio_ext)
    else:
        class FormatError(Exception):
            print('NOT supported file format')
        raise FormatError()

    return audio_file_name
