FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Mise à jour avec les versions compatibles de torch et torchvision
RUN pip install --no-cache-dir \
    numpy==1.22.0 \
    pillow==9.2.0 \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    --find-links https://download.pytorch.org/whl/torch_stable.html


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p static/upload static/processed app/models

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "2", "--timeout", "120", "--log-file", "-"]
