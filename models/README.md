# Real-Time Text & Image Prediction API

A Dockerized FastAPI service that accepts text or images and returns ML predictions as JSON.

## 1. Train the demonstration models

```bash
python train_models.py
```

## 2. Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

http://localhost:8000/docs

## 3. Test text prediction

```bash
curl -X POST "http://localhost:8000/predict/text" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"This product is excellent\"}"
```

## 4. Test image prediction

The demonstration image model expects a handwritten-digit-style image.

```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@digit.png"
```

## 5. Docker

```bash
docker build -t realtime-ml-api .
docker run --rm -p 8000:8000 realtime-ml-api
```

For Render, the service uses the PORT environment variable automatically.

## Production improvements

- Replace the demonstration models with models trained on your actual dataset.
- Add authentication/API keys if the endpoint is not meant to be public.
- Add rate limiting and request tracing.
- Add model versioning and monitoring.
- Use a dedicated model-serving worker for large deep-learning models.
