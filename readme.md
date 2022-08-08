# Project Name
Bert Based Sentiment Model for health and fitness App

![Header](https://github.com/JoAmps/bert-based-health-and-Fitness-sentiment-system/blob/main/images/header.png)


## Project Intro/Objective
Every Individual has their views and opinions on the apps they use, such views and opinions ultimately decide if they would like or dislike the app and if they would continue to use the app. For a company to be able to identify if the sentiments of the individuals regarding their products are positive or negative is crucial, so they know if the app is solving what it's intended to solve and if the customers love using the app. If most sentiments are negative, then the company knows there's something wrong, and immediate action is taken to resolve it, even if the sentiments are mostly positive, which gives a good feeling to the creators of the health apps, it encourgaes them to keep improving on the app to better serve the customers, so a way to identify if the reviews left by users are positive or negative is therefore crucial to the survival of these health and fitness apps

The figure below shows the fitness apps used in this project,
![Apps](https://github.com/JoAmps/bert-based-health-and-Fitness-sentiment-system/blob/main/images/apps.png)

### Methods Used
* Data exploration/descriptive statistics
* Data processing/cleaning
* API
* Data Version control
* Model Packaging
* Backend Development
* Data Visualization
* Deep Learning
* Experiment tracking
* Testing
* Deployment
* Containerization
* Continous integration/Continous deployment(CI/CD)
* Cloud data storage
* Cloud Container registry
* Cloud Container service
* Prediction Monitoring

### Technologies
* Python
* Google play store api
* Pytorch for deep learning learning
* Bert model - Transformers from hugging face
* ONNX for model packaging
* Docker for containerization
* Neptune for experiment tracking
* Flask for building the API
* Visual studio code, jupyter
* Git
* Unit Testing(pytest)
* Github actions for CI/CD
* AWS S3 for storing files on the cloud
* AWS EC2 for launching virtual servers
* AWS Elastic container registry for hosting docker images on the cloud
* AWS Elastic container service for deploying model
* Kibana for monitoring the deep learning model predictions on the cloud

## Project Description

#### The ability to predict and know the sentiments of the users of an app is very important, so the company heads or app creators can know the necessary steps to take to make the user experience better to keep the users using the app and to attract more users to the app since one component people use to join an app is the reviews on the app. Health and fitness are very important to everyone, those trying to lose weight, those trying to get lean and tone up, those trying to live a better life, so a lot of users are using these apps, and at the same time, a lot of apps are available that if one app has bad reviews or the sentiments on those apps is mostly negative, users quickly download other apps to use. One of the popular apps, myfitnesspal, has a lot of positive sentiments, so users keep coming back. It is therefore very important to be able to predict the sentiments of users based on their reviews. 
#### The data was scraped from the google play store on 30 health and fitness apps using the google play store API

### Some of the questions and challenges encountered were:
#### Since the bert model is big, how best to deploy it to the cloud
#### How to track all the metrics during training in realtime
#### How to make this model accessible to users
#### How to monitor the performance of the model in real time , as its been used by users


## Getting Started
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. The data folder contains code used to pull data from the google play store API and since that code has been run, the actual data used can be found in the folder also
3. The images folder contains screenshots taken of training and loss curves, confusion matrix, system metrics, monitoring dashboards and etc
4. Model folder contains code used to prepare, train, and evaluate the model
From the homepage, 

5. Requirements.txt file contains all the libraries and dependencies used for this project
6. Docker file contains steps and instructions to containerize the application so it can be easily deployed to the cloud
7. Docker-compose.yml contains code to convert the Dockerfile into a format for ease of use to build and run the container
8. Final BERT.ipynb contains the exploratory data analysis used
9. train.py contains code that strings the code from other python scripts together to do the actual training, evaluation and saving of the model and metrics


## Results
### Model Metrics
As mentioned previously, the model used was the bert model architecture, which was finetuned using Pytorch on the health and fitness dataset, which contained over 20000 reviews, with almost an equal distribution of negative and positive sentiments. The data was split into 80% for training and 20% for validation. Some preprocessing was performed, primarily using the bert tokenizer API from hugging face. The model was then trained and evaluated using the metrics, the key metric used during training was the accuracy score. Since the data was balanced, the accuracy score was a good metric to choose. The figure below shows the loss and accuracy curve of the training and validation sets.
![Loss and accuracy graphs](https://github.com/JoAmps/bert-based-health-and-Fitness-sentiment-system/blob/main/images/loss_accuracy%20graphs.png)


The model was evaluated on the roc curve, which maps the true positive rate against the false positive rate, and it achieved an 80% AUC score. The roc curve can be seen in the figure below.

![ROC curve](https://github.com/JoAmps/bert-based-health-and-Fitness-sentiment-system/blob/main/images/roc_curve.png)

The model was also evaluated using the confusion matrix which gives more information on the performance of the model, in total there were 1047 reviews in the validation set, with 250 of them incorrectly predicted(almost 24%) by the model, 125 each of false positive and false negative. Both negative predictions and positive predictions are equally important, so there is no need to change the thresholds to try to reduce false negatives or false positives. The figure below shows the confusion matrix,

![Confusion matrix](https://github.com/JoAmps/bert-based-health-and-Fitness-sentiment-system/blob/main/images/confusion%20matrix.png)

The system metrics recorded during training and logged using Neptune can be seen below. Neptune was used to track all the experiments and all the metrics so they can be reproduced

![System metrics](https://github.com/JoAmps/bert-based-health-and-Fitness-sentiment-system/blob/main/images/sys%20metrics.png)


### Deployment 
After getting a good enough model, the model needed to be deployed but first needed to create an API so it can be accessed by users when finally deployed and served to them. Flask was used as the API and the application was put in a container using docker. Before then, since the bert model is very big, and pushing it around to Github and the likes, I needed to version control the model, store it in a remote repository somewhere so I can access it anytime I want, so I put it in AWS s3 via data version control(DVC). The dockerized application was pushed to the cloud(AWS) via the elastic container registry, and then deployed using the elastic container service and exposed for users to use.
A sample of the API exposed using the AWS ECR using the postman service, for both positive and negative sentiments can be seen in the figures below,

Positive sentiment



![positive sentiment](https://github.com/JoAmps/bert-based-health-and-Fitness-sentiment-system/blob/main/images/positve%20sentiment%20.png)

Negative sentiment


![Negative sentiment](https://github.com/JoAmps/bert-based-health-and-Fitness-sentiment-system/blob/main/images/negative%20sentiment.png)

 The logs from the cloud watch service is then 




# Conclusion
Predicting the sentiment of users using the health and fitness apps in the google play store is very important to companies and app creators. Knowing these sentiments helps them understand how their customer base reacts to their apps, so they can make the required and appropriate changes to improve user experience. Sometimes the overall sentiments on the app can be good, but one update that the users don't like, can cause the sentiments to change, so when management sees this negative sentiment they can do the needful and revert that update made. Knowing the sentiment of people and reacting to it can be a very crucial component that keeps an app ahead of its competitors to keep users from churning and keep the profit coming.
