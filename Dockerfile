# Single image used for both the FastAPI server and the Streamlit demo --
# docker-compose.yml runs it twice with a different CMD override for each,
# so there's one dependency set to keep in sync instead of two.
FROM python:3.11-slim

WORKDIR /app

# System deps: only what pyarrow/pandas wheels occasionally need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Neither service needs both ports open, but exposing both keeps the image
# usable for either role without rebuilding.
EXPOSE 8000 8501

# Default: API server. docker-compose.yml overrides `command:` for the
# streamlit service. Running this image standalone (`docker run <image>`)
# still does something sensible.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
