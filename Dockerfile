
FROM python:3.11-slim
WORKDIR /application
COPY . /application/

RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
CMD ["python3", "app.py"]
