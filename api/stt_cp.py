import torch
from utils_stt import Decoder, read_batch, prepare_model_input

def main_stt(args, **kwargs):
    logger = kwargs.get("logger")
    device = kwargs.get("device")

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
