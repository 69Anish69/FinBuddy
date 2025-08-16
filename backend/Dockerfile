FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app/
EXPOSE 8000
# show we can read a secret (optional)
# ENV FINBUDDY_API_KEY_FILE=/run/secrets/finbuddy_api_key
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
