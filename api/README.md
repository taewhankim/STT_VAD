
# VAD

**Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier.**
Enterprise-grade Speech Products made refreshingly simple (see our [STT](https://github.com/snakers4/silero-models) models).

## Getting Started

The models are small enough to be included directly into this repository. Newer models will supersede older models directly.

**Currently we provide the following endpoints:**

| model=                        | Params | Model type          | Streaming | Languages      | PyTorch | ONNX | Colab |
|--------------------------------|--------|---------------------|--------------------|----------------|---------|------|-------| 
| `'silero_vad'`             | 1.1M   | VAD                 |  Yes       | `ru`, `en`, `de`, `es` (*) | :heavy_check_mark: | :heavy_check_mark:      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_vad_micro'`             | 10K   | VAD                 |  Yes       | `ru`, `en`, `de`, `es` (*) | :heavy_check_mark: | :heavy_check_mark:      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_vad_micro_8k'`             | 10K   | VAD                 |  Yes       | `ru`, `en`, `de`, `es` (*) | :heavy_check_mark: | :heavy_check_mark:      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_vad_mini'`             | 100K   | VAD                 |  Yes       | `ru`, `en`, `de`, `es` (*) | :heavy_check_mark: | :heavy_check_mark:      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_vad_mini_8k'`             | 100K   | VAD                 |  Yes       | `ru`, `en`, `de`, `es` (*) | :heavy_check_mark: | :heavy_check_mark:      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_number_detector'` | 1.1M   | Number Detector     | No       | `ru`, `en`, `de`, `es` | :heavy_check_mark: | :heavy_check_mark:      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |
| `'silero_lang_detector'`   | 1.1M   | Language Classifier |  No       | `ru`, `en`, `de`, `es` | :heavy_check_mark: | :heavy_check_mark:      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/snakers4/silero-vad/blob/master/silero-vad.ipynb) |


### dependencies


We are keeping the colab examples up-to-date, but you can manually manage your dependencies:

- `pytorch` >= 1.7.1 (there were breaking changes in `torch.hub` introduced in 1.7);
- `torchaudio` >= 0.7.2 (used only for IO and resampling, can be easily replaced);
- `soundfile` >= 0.10.3 (used as a default backend for torchaudio, can be replaced);

All of the dependencies except for PyTorch are superficial and for utils / example only. You can use any libraries / pipelines that read files and resample into 16 kHz.
