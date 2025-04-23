FROM python:3.9-slim

WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install pip requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

EXPOSE 5000

CMD ["python", "main.py"]

