FROM python:3.8
WORKDIR /usr/src/app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY ./src/requirements.txt ./
RUN pip install -r requirements.txt
COPY src .
ENV PYTHONUNBUFFERED=True \
    PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0",  "--port" "8000", "--reload"]