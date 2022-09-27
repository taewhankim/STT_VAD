import argparse
import logging
from pathlib import Path
from vad import main_vad
from stt import main_stt
from define import *
import os
from utils_vad import *
from timeit import default_timer as timer
from datetime import timedelta
from logger import get_logger
from utils import post_processing

def get_args_parser():
    p = argparse.ArgumentParser('speech to text', add_help=False)
    p.add_argument('--lang', type=str, default='NOT_A_LANG')
    p.add_argument('--input_fn', type=str, default='files/en.wav')
    p.add_argument('--timestamps', action="store_true", required=False)
    p.add_argument('--output_dir', type=str, default='output/')
    p.add_argument('--sample_rate', type=int, default=16000)
    p.add_argument('--speech_interval', type=float, default=0.3, help='second')

    arg = p.parse_args()
    return arg


if __name__ == '__main__':
    start = timer()
    logger = get_logger(log_level=logging.DEBUG)

    args = get_args_parser()
    args.vad_model = vad_model

    # minimum length(second) of words. == min_speech_samples in utils_vad
    # TODO ?
    args.min_leng_word = 10000

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    args.timestamps , wav = main_vad(args, logger=logger)

    save_audio('/mnt/dms/prac/only_speech.wav',
               collect_chunks(args.timestamps, wav), 16000)


    flg = True
    if args.lang in stt_languages:
        lang_id = [idx for idx, lang in enumerate(stt_languages) if lang == args.lang]
        args.stt_model = stt_model[lang_id[0]]
        output = main_stt(args, logger=logger)
        flg = False
        logger.debug(f"{lang_id[0]} model is loads")
        output = main_stt(args, logger=logger)

    results = post_processing(args, output[0])

    out_fn = os.path.join(args.output_dir, f'{Path(args.input_fn).stem}_speech_timestamps.txt')

    with open(out_fn, 'w') as f:
        for line in args.timestamps:
            if 'text' in line.keys() and line['text']:
                f.write(f"{str(line['start'])}-{str(line['end'])}-{line['text']}\n")
            elif flg:
                f.write(f"{str(line['start'])}-{str(line['end'])}\n")

    end = timer()

    print(timedelta(seconds=end-start))


