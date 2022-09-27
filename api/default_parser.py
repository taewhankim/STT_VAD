
import argparse

def get_args_parser():
    p = argparse.ArgumentParser('speech to text', add_help=False)
    p.add_argument('--lang', type=str, default='NOT_A_LANG')
    p.add_argument('--input_fn', type=str, default='files/en.wav')
    p.add_argument('--timestamps', action="store_true", required=False)
    p.add_argument('--sample_rate', type=int, default=16000)
    p.add_argument('--speech_interval', type=float, default=0.3, help='second')

    args, unknown = p.parse_known_args()
    return args