# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch import Tensor
from pathlib import Path

from py_series.py_hangul.hanspell import spell_checker
from py_series.py_space.pykospacing.kospacing import Spacing

from korean_stt.kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from korean_stt.kospeech.data.audio.core import load_audio, load_audio2

# 수정 : args 추가
def parse_audio(audio_path: str, del_silence: bool = False) -> Tensor:
    # 수정 : args 추가
    signal = load_audio(audio_path, del_silence, extension=Path(audio_path).suffix)

    assert list(signal), 'DO NOT LOAD SIGNAL'

    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)

# 수정
def parse_audio2(args,audio_path: str, del_silence: bool = False) -> Tensor:
    # 수정 : args 추가
    signal = load_audio2(args, audio_path, del_silence, extension=Path(audio_path).suffix)

    assert list(signal), 'DO NOT LOAD SIGNAL'
    ## 수정
    # feature = torchaudio.compliance.kaldi.fbank(
    #     waveform=Tensor(signal).unsqueeze(0),
    #     num_mel_bins=80,
    #     frame_length=20,
    #     frame_shift=10,
    #     window_type='hamming'
    # ).transpose(0, 1).numpy()
    #
    # feature -= feature.mean()
    # feature /= np.std(feature)
    #
    # return torch.FloatTensor(feature).transpose(0, 1)
    result = []
    for i in signal:
        feature = torchaudio.compliance.kaldi.fbank(
            waveform=Tensor(i).unsqueeze(0),
            num_mel_bins=80,
            frame_length=20,
            frame_shift=10,
            window_type='hamming'
        ).transpose(0, 1).numpy()

        feature -= feature.mean()
        feature /= np.std(feature)
        result.append(torch.FloatTensor(feature).transpose(0, 1))

    return result



def remove_duplicate(sentences):
    result = list()
    for s in sentences:
        pre_char = ''
        edit_sentence = ''
        for cur_char in s:
            if pre_char != cur_char:
                edit_sentence += cur_char
            pre_char = cur_char
        result.append(edit_sentence)
    return result


def ko_main(audio_path, device):
    model_path = 'stt_models/model.pt'
    feature = parse_audio(audio_path, del_silence=True)

    input_length = torch.LongTensor([len(feature)])
    vocab = KsponSpeechVocabulary('korean_stt/vocab/aihub_labels.csv')

    model = torch.load(model_path, map_location=lambda storage, loc: storage).to(device)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    model.device = device
    y_hats = model.recognize(feature.unsqueeze(0), input_length)

    ## 수정
    sentence_list = vocab.label_to_string(y_hats.cpu().detach().numpy())
    first_sent = remove_duplicate(sentence_list)
    first_sent = first_sent[0]
    spacing = Spacing()
    new_sent = first_sent.replace(" ", '')
    kospacing_sent = spacing(new_sent)
    spelled_sent = spell_checker.check(kospacing_sent)

    hanspell_sent = spelled_sent.checked
    outputs = hanspell_sent
    print(outputs)

    #print(remove_duplicate(sentence_list)[0])
    # return remove_duplicate(sentence_list)
    return outputs

## 수정 : 추가
def ko_main2(args, audio_path, device):
    model_path = 'stt_models/model.pt'

    vocab = KsponSpeechVocabulary('korean_stt/vocab/aihub_labels.csv')

    model = torch.load(model_path, map_location=lambda storage, loc: storage).to(device)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    model.device = device

    sentence_list = []
    feature = parse_audio2(args, audio_path, del_silence=True)
    ## 수정
    # for i in feature:
    #     input_length = torch.LongTensor([len(i)])
    #     y_hats = model.recognize(i.unsqueeze(0), input_length)
    #     sentence_list.append(remove_duplicate(vocab.label_to_string(y_hats.cpu().detach().numpy())))

    for i in feature:
        input_length = torch.LongTensor([len(i)])
        y_hats = model.recognize(i.unsqueeze(0), input_length)

        first_sent = remove_duplicate(vocab.label_to_string(y_hats.cpu().detach().numpy()))
        first_sent = first_sent[0]
        spacing = Spacing()
        new_sent = first_sent.replace(" ", '')
        kospacing_sent = spacing(new_sent)
        spelled_sent = spell_checker.check(kospacing_sent)

        hanspell_sent = spelled_sent.checked
        outputs = hanspell_sent

        sentence_list.append(outputs)

    #print(remove_duplicate(sentence_list)[0])

    return sentence_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Korean Speech Recognition')
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--device', type=str, required=False, default='cpu')
    opt = parser.parse_args()

    sentence = ko_main(opt.audio_path, opt.device)
