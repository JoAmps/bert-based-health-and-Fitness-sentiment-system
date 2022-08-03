import warnings
warnings.filterwarnings("ignore")
from transformers import BertTokenizer
import torch
from healthcheck import HealthCheck
from model.model import BertClassifier
from model.prepare_data import preprocessing_for_bert
from flask import Flask, jsonify, request
import os


app = Flask(__name__)
health = HealthCheck(app, "/hcheck")



def howami():
    return True, "I am alive. Thanks for checking.."

health.add_check(howami)



def configure_model(path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bert_classifier = BertClassifier(freeze_bert=True)
    bert_classifier.load_state_dict(torch.load(path, map_location=device))
    print(device)
    #bert_classifier.to(device)
    return bert_classifier



#def configure_data(review_text):
 #   bert_classifier=configure_model("model_weights.pt")
  #  input_ids, attention_masks = preprocessing_for_bert(review_text)
   # output = bert_classifier(input_ids, attention_masks)
    #_, prediction = torch.max(output, dim=1)
    #return prediction


#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#model = Net()
#model.load_state_dict(torch.load(weights_paths, map_location=device))
# Or you can move the loaded model into the specific device
#model.to(device)

def configure_data(review_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
    encoded_review = tokenizer.encode_plus(
    review_text,
    max_length=256,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
    )
    bert_classifier=configure_model("model_weights.pt")

    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    output = bert_classifier(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    
    return prediction


@app.route("/predict")
def predict():
    review_text = request.args.get("text")
    class_names=['negative','positive']
    prediction = configure_data(review_text)
    response = {}
    response["response"] = {
        'REVIEW':review_text,'SENTIMENT':class_names[prediction]}
    
    return jsonify(response)    

@app.route('/')
def hello():
    return 'Welcome to Health and fitness Sentiment App '

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    