# 어떤 이미지를 사용할건지 선택
FROM nvidia/cuda:10.2-cudnn7-devel

# Time zone setting 시 발생하는 문제 해결
ENV DEBIAN_FRONTEND noninteractive

# RUN 은 명령어를 실행하는 도커 문법(명령어)
RUN apt-get update && apt-get install -y \
        ca-certificates python3-dev git wget sudo vim openssh-server \
        cmake ninja-build protobuf-compiler libprotobuf-dev locales net-tools \
        libgl1-mesa-glx tzdata libsndfile1 libavcodec-extra ffmpeg iputils-ping curl && \
        rm -rf /var/lib/apt/lists/*

# settings Asia timezone
ENV TZ Asia/Seoul

# UTF-8 사용할 수 있게 설정
RUN localedef -f UTF-8 -i ko_KR ko_KR.UTF-8
RUN export LC_ALL=ko_KR.UTF-8
ENV LANG ko_KR.UTF-8
ENV LANGUAGE ko_KR.UTF-8
RUN update-locale LANG=ko_KR.UTF-8

# python symbolic link 생성
RUN ln -sv /usr/bin/python3 /usr/bin/python

# ENV는 환경변수를 설정하는 도커 문법(명령어)
ENV PATH=/root/.local/bin:${PATH}

RUN wget https://bootstrap.pypa.io/get-pip.py && \
        python3 get-pip.py --user && \
        rm get-pip.py

ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# 이렇게 해야 차후 코드만 변경됐을때, 패키지 설치 없이 코드만 반영되서 도커 이미지가 만들어짐
COPY requirements.txt /api/requirements.txt
COPY korean_stt /api/korean_stt

WORKDIR /api
RUN pip install -r requirements.txt
RUN pip install -e /api/korean_stt

COPY . /api


# -b 뒤에 숫자는 docker-compose.yml 파일에서 api services 내에 expose, ports 와 동일하게 적용
CMD ["gunicorn", "-w", "3", "-b", ":9003", "-t", "360", "--reload", "wsgi:app"]
