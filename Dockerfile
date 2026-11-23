FROM python:3.11-slim

WORKDIR /app

# Serving-only dependencies (no dvc / mlflow / optuna needed to run the app)
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

# Run `dvc pull` (or `dvc repro`) BEFORE `docker build` so models/model.pkl exists.
COPY app.py .
COPY models/model.pkl ./model.pkl
COPY data/raw/bank.csv ./bank.csv
COPY reports/ ./reports/

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
