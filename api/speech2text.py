
import flask
from utils import post_processing, init_args, wrap_data
from default_parser import get_args_parser
from vad import get_time_stamps
from stt import main_stt
from logger import get_logger
import logging
import json
from define import *
from timeit import default_timer as timer
from datetime import timedelta
import torch
app = flask.Flask(__name__)

@app.route("/speech2text", methods=["POST","GET"])
def service():
    # Could not run 'aten::empty_strided' with arguments from the 'QuantizedCUDA' backend ERROR 로 인하여, cuda 는 X
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    start = timer()

    logger = get_logger(name="speech2text", log_level=logging.DEBUG)

    request_data = None

    if flask.request.get_data():
        request_data = flask.request.form
        # lang = flask.request.form.get('language')
        # path2 = flask.request.form.get('path')
        # request_data = {}
        # request_data['language'] = lang
        # request_data['path'] = path2
    else:
        request_data = json.loads(flask.request.data.decode("utf-8"))
    logger.debug(f"request_data: {request_data}")
    args = init_args(get_args_parser(), language=request_data.get("language"), path=request_data.get("path"))

    dash = "*" * 50
    logger.debug(f"{dash}")
    logger.debug(f"initialize args from request data: {args}")

    # 수정
    args.timestamps = True
    if args.timestamps:
        args.timestamps = get_time_stamps(args, logger=logger, default_adaptive_parameters=default_adaptive_parameters, device=device)

        # 수정 : enumerate , idx 추가
        # for idx, timestamp in enumerate(args.timestamps):
        if args.lang in stt_languages:
            lang_id = [idx for idx, lang in enumerate(stt_languages) if lang == args.lang]
            args.stt_model = stt_model[lang_id[0]]
            logger.debug(f"{lang_id[0]} model is loads")
            ## 수정
            # _ = main_stt(args, logger=logger, device=device) # 시간단위 제외한 결과값, 실제 사용하는 값은 args.timestamps 에 있는 값을 사용
            total_text = main_stt(args, logger=logger, device=device)  # 시간단위 제외한 결과값, 실제 사용하는 값은 args.timestamps 에 있는 값을 사용
        text = post_processing(args,total_text)
    else:
        if args.lang in stt_languages:
            lang_id = [idx for idx, lang in enumerate(stt_languages) if lang == args.lang]
            args.stt_model = stt_model[lang_id[0]]
            logger.debug(f"{lang_id[0]} model is loads")
            ## 수정
            # _ = main_stt(args, logger=logger, device=device) # 시간단위 제외한 결과값, 실제 사용하는 값은 args.timestamps 에 있는 값을 사용
            text = main_stt(args, logger=logger, device=device)

    ## 수정 : 주석

    outputs = wrap_data(request_data, text)
    end = timer()
    logger.debug(f"processing time: {timedelta(seconds=end-start)}")
    logger.debug(f"{dash}\n")
    return flask.jsonify(outputs)

@app.route("/speech2texttest", methods=["POST","GET"])
def test():
    outputs = {
        "path": "/mnt/aistudio/dms",
        "langauge": "en",
        "result": "success"
               }

    return flask.jsonify(outputs)

if __name__ == '__main__':
    #app.run(host="0.0.0.0")
    #app.run(host="127.0.0.1", port=9002)
    app.run(host="0.0.0.0", port=9005)
