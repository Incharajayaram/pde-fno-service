FROM python:3.11-slim

WORKDIR /app

# CPU-only torch build (Render has no GPU, and only inference runs here) --
# keeps the image small compared to the default CUDA-bundled PyPI wheel.
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/

# No dataset, no training scratch -- only the small deployment checkpoint
# under app/model/ is needed at runtime.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
