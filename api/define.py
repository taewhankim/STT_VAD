# stt
stt_model_root = 'stt_models'
stt_languages = ['en', 'de', 'es', 'ua', 'ko']
stt_model = [
    'stt_models/en_v3_jit.model',
    'stt_models/de_v1_jit.model',
    'stt_models/es_v1_jit.model',
    'stt_models/ua_v3_jit.model',
    'ko'
]

# vad
vad_model = 'vad_models/model.jit'

default_adaptive_parameters = {
    "batch_size": 200,
    "step": 500,
    "speech_interval": 0.3,
    "sample_rate": 16000, # default
    "num_samples_per_window": 4000,
    "min_speech_samples": 10000,
    "min_silence_samples": 6000,
    "speech_pad_samples": 2000,
}
