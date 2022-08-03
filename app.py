from flask import Flask, jsonify, request
#from preprocessing.functions import tokenize
import xgboost as xgb
import joblib
from healthcheck import HealthCheck
import os
import logging
from model.model import BertClassifier, initialize_model
from model.prepare_data  import text_preprocessing, preprocessing_for_bert, create_data_loaders

import torch
#import onnxruntime as onx
#app = Flask(__name__)

#target={0:'Negative', 1:'Positive'}

#model = joblib.load('models/model.onxx')
#print('done')
class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        bert_classifier = BertClassifier(freeze_bert=True)
        bert_classifier.load_state_dict(torch.load(model_path,map_location='cpu'))
        self.model=bert_classifier.eval()
        #self.model=bert_classifier.freeze()
        #bert_classifier.load_state_dict(torch.load(path))
        #self.model=torch.load(model_path)
        #self.model.eval()
        #
        self.processor = create_data_loaders()
        self.softmax = torch.nn.Softmax(dim=1)
        self.labels = ["Negative", "Positive"]

    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]),
            torch.tensor([processed["attention_mask"]]),
        )
        scores = self.softmax(logits[0]).tolist()[0]
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "This is a very lovely app, excellent by all standards"
    predictor = ColaPredictor("./model/model_weights.pt")
    print(predictor.predict(sentence))
    #sentences = ["The boy is sitting on a bench"] 
    #for sentence in sentences:
    predictor.predict(sentence)