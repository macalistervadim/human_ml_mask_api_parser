FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG MODEL_ID=1ruJg4lqR_jgQPj-9K0PP-L2vJERYOxLP
ARG MODEL_NAME=exp-schp-201908301523-atr.pth

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN pip install --no-cache-dir gdown \
    && mkdir -p /app/schp

RUN if [ ! -f "/app/${MODEL_NAME}" ]; then \
        echo "Downloading model ${MODEL_NAME} (id=${MODEL_ID})"; \
        gdown --id "${MODEL_ID}" -O "/app/${MODEL_NAME}"; \
    else \
        echo "Model already exists, skipping download."; \
    fi

RUN ln -sf "/app/${MODEL_NAME}" "/app/schp/${MODEL_NAME}"

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
