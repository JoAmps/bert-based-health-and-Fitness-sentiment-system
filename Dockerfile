FROM huggingface/transformers-pytorch-gpu:latest

COPY requirements.txt .

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
