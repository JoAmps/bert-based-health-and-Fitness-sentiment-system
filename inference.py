import warnings
warnings.filterwarnings("ignore")
from transformers import BertTokenizer
import torch
from healthcheck import HealthCheck
from model.model import BertClassifier
from flask import Flask, jsonify, request
import os


app = Flask(__name__)
health = HealthCheck(app, "/hcheck")



def howami():
    return True, "I am alive. Thanks for checking.."

health.add_check(howami)



def configure_model(path):
    bert_classifier = BertClassifier(freeze_bert=True)
    bert_classifier.load_state_dict(torch.load(path))
    return bert_classifier



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
    bert_classifier=configure_model("model\model_weights.pt")

    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    output = bert_classifier(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    #
    return prediction


#@app.route('/predict', methods=['GET'])
#def predict():
  #  return 'Hello World!'

#@app.route('/predict')
#def predict():
  #  data = request.json()
  #  review_text = data['text']
  #  class_names=['negative','positive']
  #  prediction = configure_data(review_text)
  #  return jsonify({'REVIEW':review_text,'SENTIMENT':class_names[prediction]})
    #print(f'Review text: \n\n {review_text}')
    #print('-'*80)
    #print(f'Sentiment  : \t {class_names[prediction]}')
    

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
    #.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))