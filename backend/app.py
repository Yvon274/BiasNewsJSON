import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import ssl

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'temp4.json')

# # Assuming your JSON data is stored in a file named 'init.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)
    articles_df = pd.DataFrame(data['articles'])
    
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()
left_keywords = ['liberal', 'progressive', 'socialist', 'equality', 'justice', 'Biden', 'Ukraine']
right_keywords = ['conservative', 'capitalist', 'free market', 'individual freedom', 'traditional values', 'Trump']

def determine_political_leaning(text):
    # Sentiment analysis
    sentiment_score = sid.polarity_scores(text)['compound']
    
    # Named entity recognition (NER) found this on the internet and seemed useful for the future
    # doc = nlp(text)
    # entities = [ent.text.lower() for ent in doc.ents]
    
    # Keyword analysis
    tokens = text.lower().split()
    left_score = sum(token in left_keywords for token in tokens)
    right_score = sum(token in right_keywords for token in tokens)
    
    # Assign weights to each component in the future this can be changing but for now it is hardcoded
    sentiment_weight = 0.7
    keyword_weight = 0.3
    
    #if it is positive then I just assumed that the one with more terms said is the one that it is being positive about
    if right_score + left_score == 0:
      return 0

    if sentiment_score >= 0:
      if right_score >= left_score:
        keywords = (right_score - left_score)/(right_score+left_score)
      else:
        keywords = (left_score - right_score)/(right_score+left_score)
      result = sentiment_score * sentiment_weight + keywords * keyword_weight
    else:
      if right_score >= left_score:
        keywords = (left_score - right_score)/(right_score+left_score)
      else:
        keywords = (right_score - left_score)/(right_score+left_score)
      result = sentiment_score * sentiment_weight + keywords * keyword_weight
    return result
articles_df['score'] = articles_df['text'].apply(determine_political_leaning)


app = Flask(__name__)
CORS(app)

# Sample search using json with pandas
def json_search(query):
    matches = articles_df[articles_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'text', 'score']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/articles")
def articles_search():
    text = request.args.get("title")
    return json_search(text)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)














