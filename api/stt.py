import torch
from utils_stt import Decoder, read_batch, prepare_model_input
from korean_stt.inference import ko_main, ko_main2
from logger import get_logger
from utils import init_args
from default_parser import get_args_parser
import logging
from define import *
import sys


def main_stt(args, **kwargs):
    logger = kwargs.get("logger")
    device = kwargs.get("device")

    ## 수정
    if not args.timestamps:
        if args.stt_model == 'ko':
            outputs = ko_main(args.input_fn, device)
            return outputs

    ## 여기까지 수정
    else:
        if args.stt_model == 'ko':
            outputs = ko_main2(args,args.input_fn, device)
            return outputs

    batch = read_batch([args.input_fn], args.sample_rate)
    input = prepare_model_input(batch, device=device)

    model = torch.jit.load(args.stt_model, map_location=device)
    logger.debug(f"load {args.stt_model} model")

    decoder = Decoder(model.labels)
    logger.debug(f"load decoder model")

    outputs = []

    for t in args.timestamps:
        in_tensor = input[:, t['start']:t['end']]
        wav_len = in_tensor.shape[1] / args.sample_rate

        model.eval()
        output = model(in_tensor)

        output_text = decoder(output[0].cpu(), wav_len, word_align=True)[-1]

        t['text'] = output_text
        outputs.append(output_text)
    logger.debug(f"inference is done")
    return outputs



if __name__ == '__main__':
    device = torch.device('cpu')

    logger = get_logger(name="speech2text", log_level=logging.DEBUG)
    args = init_args(get_args_parser(), language='ko', path='korean_stt44/vocab/demo_audio/bM32LJFpAus.0282.wav')
    args.stt_model = stt_model[4]
    print(main_stt(args, logger=logger, device=device))

