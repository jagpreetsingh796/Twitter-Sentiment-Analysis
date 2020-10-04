import tweepy
import  streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


consumer_key="OfryBtdLqeomROavJLAgalBlg"
consumer_secret="FPTwbBWlRgyFbW8bx7bofnNQUfrcikRKIBrypaqvvBwIOUizLQ"
access_token="1270158711357607943-vZsmjH9wYJrQ0us8pCefQkIgXgzX8S"
access_token_secret="MPMqEOzHsuqWmTMp1UKw9x81lLzVcQK8Y1jbJzcMpnpiA"

auth=tweepy.AppAuthHandler(consumer_key,consumer_secret)
# auth.set_access_token(access_token,access_token_secret)
api=tweepy.API(auth,wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

Analyzer_name=st.sidebar.selectbox("Select Analyzer",("TextBlob","Vader"))
print("Analyzer name {}".format(Analyzer_name))
term=st.text_input("Twitter Search Term","Enter here")
num=st.number_input("Number of Tweets",min_value=0,value=0)
li=[]
result={}




def get_tweets(term,num):
  tweets=tweepy.Cursor(api.search,q=term,language="English").items(num)
  corpus=[]
  for tweet in tweets:
    no_links = re.sub(r'http\S+', '', tweet.text)
    no_unicode = re.sub(r"\\[a-z][a-z]?[0-9]+", '', no_links)
    tweet= re.sub('[^A-Za-z ]+', '', no_unicode)

    # tweet=re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet.text)
    corpus.append(tweet)


  return corpus



def choose_model(name):
  if(name=='TextBlob'):
    di=perform_analysis_textBlob(li)
    labels = ['Negative', 'Weak Negative', 'Strong Negative', 'Positive', 'Weak Positive', 'Strong Positive', 'Neutral']
    values = [di['Negative'], di['Weak Negative'], di['Strong Negative'], di['Positive'], di['Weak Positive'],
              di['Strong Positive'], di['Neutral']]

    fig=go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    st.plotly_chart(fig)



    

  else:
    di=perform_analysis_vader(li)
    labels = ['Negative', 'Positive', 'Neutral']
    values = [di['Negative'], di['Positive'], di['Neutral']]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])

    st.plotly_chart(fig)

def percentage(part,whole):
  return (part/whole) *100


def perform_analysis_textBlob(result_set):
  polarity = 0.0
  positive = 0.0
  spositive = 0.0
  wpositive = 0.0
  negative = 0.0
  snegative = 0.0
  wnegative = 0.0
  neutral = 0.0


  for tweet in result_set:

    ob = TextBlob(tweet)
    curr_polarity = ob.sentiment.polarity
    polarity += curr_polarity
    if (curr_polarity == 0.0):
      neutral += 1
    elif (curr_polarity > 0.0 and curr_polarity <= 0.3):
      wpositive += 1
    elif (curr_polarity > 0.3 and curr_polarity <= 0.6):
      positive += 1
    elif (curr_polarity > 0.6 and curr_polarity <= 1.0):
      spositive += 1
    elif (curr_polarity > -0.3 and curr_polarity < 0.0):
      wnegative += 1
    elif (curr_polarity > -0.6 and curr_polarity <= -0.3):
      negative += 1
    elif (curr_polarity > -1 and curr_polarity <= -0.6):
      snegative += 1

  negative_percentage = percentage(negative, num)
  weak_negative_percentage = percentage(wnegative, num)
  strong_negative_percentage = percentage(snegative, num)
  weak_positive_percentage = percentage(wpositive, num)
  strong_positive_percentage = percentage(spositive, num)
  positive_percentage = percentage(positive, num)
  neutral_percentage = percentage(neutral, num)

  di = {"Topic":term, "Number of tweets": num, "Weak Positive": weak_positive_percentage,
        "Positive": positive_percentage, "Strong Positive": strong_positive_percentage,
        "Weak Negative": weak_negative_percentage, "Negative": negative_percentage,
        "Strong Negative": strong_negative_percentage, "Neutral": neutral_percentage}
  return di





def perform_analysis_vader(result_set):
  positive = 0.0

  negative = 0.0

  neutral = 0.0
  analyzer = SentimentIntensityAnalyzer()


  for tweet in result_set:

    curr_polarity = analyzer.polarity_scores(tweet)

    if (curr_polarity['compound'] > -0.05 and curr_polarity['compound'] < 0.05):
      neutral += 1
    elif (curr_polarity['compound'] >= 0.05):
      positive += 1
    elif (curr_polarity['compound'] <= -0.05):
      negative += 1

  negative_percentage = percentage(negative, num)
  positive_percentage = percentage(positive, num)
  neutral_percentage = percentage(neutral, num)

  di = {"Topic":term, "Number of tweets": num, "Positive": positive_percentage, "Negative": negative_percentage,
        "Neutral": neutral_percentage}
  return di


def remove_stopwords(search_term, num):
  li = get_tweets(search_term, num)
  word = []

  for i in li:
    words = i.split(" ")
    words = [word for word in words if len(word) > 2]
    words = [word.lower() for word in words]
    words = [w for w in words if w not in STOPWORDS]
    word.extend(words)
  return (word)




if st.sidebar.button("Analyze"):
  li=get_tweets(term,num)
  choose_model(Analyzer_name)

if st.sidebar.button("Word Cloud"):
  mask=np.array(Image.open("sherlock_2.png"))
  words=remove_stopwords(term,num)
  # mask=np.ndarray((mask.shape[0],mask.shape[1]),np.int32)
  stop_words=['said']+list(STOPWORDS)

  wc = WordCloud(background_color="black",contour_width=3,mask=mask,max_words=2000,height=1100,width=1200,stopwords=stop_words)
  clean_string = ','.join(words)
  wc.generate(clean_string)
  plt.figure(figsize=(35,30))
  plt.imshow(wc,interpolation="bilinear")
  # plt.imshow(mask)

  plt.axis("off")
  st.pyplot()




