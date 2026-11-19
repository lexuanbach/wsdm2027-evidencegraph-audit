FROM python:3.12-slim
WORKDIR /audit
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY . .
CMD ["python", "revision/verify_offline.py"]
