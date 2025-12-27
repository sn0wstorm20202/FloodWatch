FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY . /app

WORKDIR /app/backend

ENV FLOODWATCH_ML_AUTO_TRAIN_ON_STARTUP=0 \
    FLOODWATCH_ML_ALLOW_API_RETRAIN=0 \
    FLOODWATCH_LOAD_ROADS_GRAPH=0

EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
