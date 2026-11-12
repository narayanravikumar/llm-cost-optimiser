# Serves the precomputed results only: no API key, no PyTorch, nothing paid per visitor.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN pip install --no-cache-dir "fastapi==0.115.*" "uvicorn[standard]==0.30.*"
COPY src/ src/
COPY web/ web/
RUN useradd --create-home app && chown -R app /app
USER app
CMD ["sh", "-c", "uvicorn src.costopt.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
