# FROM  python:3.8
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.4.3
FROM ${BASE_IMAGE} as onnxruntime
WORKDIR /usr/src/app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install python3-pip -y 
RUN pip3 install --upgrade pip
COPY ./src/requirements.txt ./
RUN pip3 install -r requirements.txt
COPY src .
ENV PYTHONUNBUFFERED=True \
    PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0",  "--port" "8000", "--reload"]