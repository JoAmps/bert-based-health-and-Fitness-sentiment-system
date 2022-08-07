FROM huggingface/transformers-pytorch-gpu:latest

FROM python:3.8

COPY ./ /app

COPY ./ /model_weights.pt

WORKDIR /app


ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY


ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY



RUN pip install "dvc[s3]"
RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('words')"

#RUN dvc init --no-scm

RUN dvc remote add -d model-store s3://healthandfitnesssentiment/trained_model/

RUN cat .dvc/config

RUN dvc pull dvc_files/trained_model.dvc

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]


#handfsentiment:latest