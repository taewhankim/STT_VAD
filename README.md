# Information
* docker compose로 실행 가능
* STT로 변환시 띄어쓰기 및 맞춤법 교정 추가
    - py_series/py_hangul
    - py_series/py_space
* timestamps 사용 여부 추가 (→ 파라미터 중  `min_silence_samples` 로 조절 가능)

# 추론

```
실행 코드 : python speech2text.py --lang ko
```


## Docker compose 배포
    * 사전에 nvidia-docker2 packages 및 docker network 생성
    * nvidia-docker2 packages 설치

        ```
        sudo apt-get update
        sudo apt-get install -y nvidia-docker2
        # restart the docker daemon to complete the installation after setting the default runtime
        ```

    * host 컴퓨터에서 docker network 생성

        ```
        docker network create speech2text_net
        ```

## Docker compose 실행
    * 실행 디렉토리 : `cd vad_korean`

        ```
        처음 배포하는 경우 : docker-compose up --build -d
        다시 배포하는 경우 : docker-compose up -d
        ```

