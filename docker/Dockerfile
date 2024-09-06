FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev build-essential \
    libssl-dev libffi-dev python3-setuptools curl

RUN pip install micropython-upip torch torchvision torchaudio snntorch numpy

COPY . /app
WORKDIR /app

ENTRYPOINT ["python", "main.py"]
