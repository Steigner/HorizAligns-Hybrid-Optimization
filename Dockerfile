FROM python:3.10

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

ENV WANDB_MODE=offline

# CMD ["python3", "scripts/main.py"]