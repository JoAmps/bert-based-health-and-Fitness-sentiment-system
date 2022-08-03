FROM huggingface/transformers-pytorch-gpu:latest

FROM python:3.8

COPY ./ /app

COPY ./ /model_weights.pt

WORKDIR /app

RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('words')"


EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]


#handfsentiment:latest