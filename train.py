from turtle import screensize
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel, BertTokenizerFast,DistilBertTokenizerFast
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.corpus import stopwords
words = set(nltk.corpus.words.words())
#from nltk.stem.porter import PorterStemmer
#from nltk.stem import WordNetLemmatizer
import re
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
pd.set_option('display.max_columns', None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from model.prepare_data  import load_data, score_to_sentiment, text_preprocessing, preprocessing_for_bert, create_data_loaders, evaluate_roc
from model.model import BertClassifier, initialize_model, set_seed, train, evaluate, bert_predict



if __name__ == '__main__':
    df=load_data('data/app_reviews.csv')
    df['sentiment_score'] = df['score'].apply(score_to_sentiment)
    df['sentiment']=df['sentiment_score'].replace({1:'negative',0:'neutral',2:'positive'})
    df=df[df['sentiment']!='neutral']
    df['sentiment_score']=df['sentiment_score'].replace({1:0,2:1})
    df=df.dropna()
    #df = df.sample(300)
    train_content, val_content,train_sentiments,val_sentiments= train_test_split(df['content'],df['sentiment_score'],test_size=0.2,random_state=RANDOM_SEED, stratify=df['sentiment'])
    train_inputs, train_masks=preprocessing_for_bert(train_content)
    val_inputs, val_masks = preprocessing_for_bert(val_content)
    train_dataloader, val_dataloader = create_data_loaders(train_sentiments, val_sentiments, train_masks,val_masks,train_inputs, val_inputs , 32)
    set_seed(42)    # Set seed for reproducibility
    bert_classifier, optimizer, scheduler = initialize_model(train_dataloader, epochs=5)
    train(bert_classifier, optimizer, scheduler, train_dataloader, val_dataloader, epochs=5, evaluation=True)
    torch.save(bert_classifier.state_dict(), 'model_weights.pt')
    path = 'model_weights.pt'
    bert_classifier.load_state_dict(torch.load(path))
    evaluate(bert_classifier, val_dataloader)
    preds=bert_predict(bert_classifier, val_dataloader)
    evaluate_roc(preds, val_sentiments)
    



    #set_seed(42)    # Set seed for reproducibility
    #bert_classifier, optimizer, scheduler = initialize_model(epochs=15)
    #train(bert_classifier, train_dataloader, val_dataloader, epochs=15, evaluation=True)
    #torch.save(bert_classifier.state_dict(), 'current_weights.pt')