FROM python:3.10-slim

# 1. Environment variables to optimize Python performance inside Docker
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/code/.cache/huggingface

WORKDIR /code

# 2. Copy dependencies list first
COPY requirements.txt .

# 3. Install CPU PyTorch + requirements without caching pip wheel files
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 4. Pre-download model weights into explicit image cache
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('marvinwong12/roberta-aita'); \
    AutoModelForSequenceClassification.from_pretrained('marvinwong12/roberta-aita')"

# 5. Copy backend application code
COPY ./app ./app
COPY helpers.py .
COPY lookup.py .

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /code
USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]