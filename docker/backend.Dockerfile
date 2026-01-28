FROM python:3.10-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ✅ COPY correct requirements path
COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ✅ Copy backend code
COPY backend /app/backend

# Create uploads directory
RUN mkdir -p /app/uploads

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
