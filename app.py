
from flask import Flask, jsonify, request
from model.prepare_data import tokenize
from model.model import BertClassifier
from healthcheck import HealthCheck
import torch
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
health = HealthCheck(app, "/hcheck")


def howami():
    return True, "I am alive. Thanks for checking.."


health.add_check(howami)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def configure_model(path):
    bert_classifier = BertClassifier(freeze_bert=True)
    bert_classifier.load_state_dict(torch.load(path, map_location=device))
    print(device)

    return bert_classifier


def configure_data(review_text):
    encoded_review = tokenize(review_text)
    bert_classifier = configure_model("model_weights.pt")
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    output = bert_classifier(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    return prediction


@app.route("/predict")
def predict():
    review_text = request.args.get("text")
    class_names = ['negative', 'positive']
    prediction = configure_data(review_text)
    response = {}
    response["response"] = {
        'REVIEW': review_text, 'SENTIMENT': class_names[prediction]}

    return jsonify(response)


@app.route('/')
def hello():
    return 'Welcome to Health and fitness Sentiment App '


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
