FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


# hazm needs Python 3.12 or 3.13 (not 3.14).
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# The CPU build first: PyPI's default torch on Linux pulls in gigabytes of CUDA libraries.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py train.py predict.py eda.py ./
COPY persian_news/ persian_news/
COPY --from=frontend /frontend/dist frontend/dist

RUN useradd --uid 1000 --create-home app && mkdir artifacts && chown app artifacts
USER app

EXPOSE 8000
HEALTHCHECK --start-period=60s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/model')"]
CMD ["python", "api.py", "--host", "0.0.0.0"]
