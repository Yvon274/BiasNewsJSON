import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from helpers.utils import QueryChecker

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import ssl

nltk.download('punkt')

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..", os.curdir))

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
left_keywords_int = [
    "Social justice",
    "Equality",
    "Diversity",
    "Inclusion",
    "Environmentalism",
    "Climate change",
    "LGBTQ",
    "Feminism",
    "Healthcare access",
    "Progressive",
    "Intersectionality",
    "Systemic racism",
    "Income inequality",
    "Workers' rights",
    "Gun control",
    "Anti-discrimination",
    "Affordable housing",
    "Universal basic income",
    "Welfare state",
    "Public education",
    "Medicare for All",
    "Indigenous rights",
    "Human rights",
    "Immigration reform",
    "Anti-war",
    "Fair trade",
    "Community organizing",
    "Grassroots activism",
    "Labor unions",
    "Renewable energy",
    "Biden",
    "Bernie Sanders",
    "Alexandria Ocasio-Cortez",
    "Noam Chomsky",
    "Angela Davis",
    "Elizabeth Warren",
    "Jeremy Corbyn",
    "Gavin Newsom"

]
right_keywords_int = [
    "Free market",
    "Individualism",
    "Limited government",
    "Traditional values",
    "Nationalism",
    "Second Amendment rights",
    "Border security",
    "Tax cuts",
    "Pro-life",
    "Religious freedom",
    "Law and order",
    "Personal responsibility",
    "Fiscal conservatism",
    "School choice",
    "Family values",
    "Military strength",
    "Patriotism",
    "Economic freedom",
    "Conservatism",
    "Capitalism",
    "National sovereignty",
    "States' rights",
    "Liberty",
    "Private property rights",
    "Self-reliance",
    "American exceptionalism",
    "Regulatory reform",
    "Entrepreneurship",
    "Traditional marriage",
    "Right to bear arms",
    "Trump",
    "Ted Cruz",
    "Tucker Carlson",
    "Ben Shapiro",
    "Candace Owens",
    "Jordan Peterson",
    "Marjorie Taylor Greene",
    "Ron Desantis"
]

left_keywords = [word.lower() for word in left_keywords_int]
right_keywords = [word.lower() for word in right_keywords_int]

# Initial weights
sentiment_weight = 0.5
keyword_weight = 0.5


def scorer(sentiment_score, text):

    tokens = text.lower().split()
    left_score = sum(token in left_keywords for token in tokens)
    right_score = sum(token in right_keywords for token in tokens)

    if right_score + left_score == 0:
        return 0
    if sentiment_score >= 0:
        if right_score >= left_score:
            keywords = (right_score - left_score)/(right_score+left_score)
            result = sentiment_score * sentiment_weight + keywords * keyword_weight
        else:
            keywords = (left_score - right_score)/(right_score+left_score)
            result = -1 * sentiment_score * sentiment_weight + keywords * keyword_weight
    else:
        if right_score >= left_score:
            keywords = (left_score - right_score)/(right_score+left_score)
            result = -1 * sentiment_score * sentiment_weight + keywords * keyword_weight
        else:
            keywords = (right_score - left_score)/(right_score+left_score)
            result = sentiment_score * sentiment_weight + keywords * keyword_weight
    return result


def determine_political_leaning(text):
    # Sentiment analysis
    sentences = sent_tokenize(text)
    total_sent = len(sentences)

    # Get the polarity of each sentence in the text
    result = []
    for sentence in sentences:
        sentiment_score = sid.polarity_scores(sentence)['compound']
        sent_score = scorer(sentiment_score, sentence)
        result.append(sent_score)

    # Return the relevant sentences that were used in making the decision
    indexes = list(enumerate(result))
    sorted_scores = sorted(indexes, key=lambda x: x[1], reverse=True)
    if sent_score <= 0:
        top_3 = [index for index, _ in sorted_scores[-3:]]
    else:
        top_3 = [index for index, _ in sorted_scores[:3]]
    relevant_sents = [(sentences[x], result[x]) for x in top_3]

    return sum(result)/total_sent


articles_df['score'] = articles_df['text'].apply(determine_political_leaning)

json_data_with_scores = {
    'articles': articles_df.to_dict(orient='records')
}

# Write JSON data with scores back to file
with open(json_file_path, 'w') as file:
    json.dump(json_data_with_scores, file, indent=4)


app = Flask(__name__)
CORS(app)

# Sample search using json with pandas


def json_search(query):
    matches = articles_df[articles_df['title'].str.lower(
    ).str.contains(query.lower())]
    matches_filtered = matches[['title', 'text', 'score', 'url']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json


def cos_search(query):
    q = QueryChecker(query)
    q.loadData('./temp4.json')

    top_indices = q.get_most_similar(query.lower(), articles_df)
    matches = articles_df.iloc[top_indices]

    matches_filtered = matches[['title', 'text', 'score', 'url']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json

def three_searchv2(query):
    q = QueryChecker(query)
    # q.loadData('./temp4.json')

    pos_articles_df = articles_df[articles_df['score'] > 1]
    neg_articles_df = articles_df[articles_df['score'] < 1]
    med_articles_df = articles_df[(articles_df['score'] >= -0.15) & (articles_df['score'] <= 0.15)]

    pos_top_indices = q.get_most_similar(query.lower(), pos_articles_df)
    neg_top_indices = q.get_most_similar(query.lower(), neg_articles_df)
    med_top_indices = q.get_most_similar(query.lower(), med_articles_df)
    all_top_indices = q.get_most_similar(query.lower(), articles_df)

    pos_matches = articles_df.iloc[pos_top_indices]
    neg_matches = articles_df.iloc[neg_top_indices]
    med_matches = articles_df.iloc[med_top_indices]
    all_matches = articles_df.iloc[all_top_indices]

    pos_matches_filtered = pos_matches[['title', 'text', 'score', 'url']]
    neg_matches_filtered = neg_matches[['title', 'text', 'score', 'url']]
    med_matches_filtered = med_matches[['title', 'text', 'score', 'url']]
    all_matches_filtered = all_matches[['title', 'text', 'score', 'url']]

    pos_matches_filtered_json = pos_matches_filtered.to_json(orient='records')
    neg_matches_filtered_json = neg_matches_filtered.to_json(orient='records')
    med_matches_filtered_json = med_matches_filtered.to_json(orient='records')
    all_matches_filtered_json = all_matches_filtered.to_json(orient='records')

    return [neg_matches_filtered_json, pos_matches_filtered_json, med_matches_filtered_json, all_matches_filtered_json]








def three_search(query):
    """
    Performs three different searches based on the query:
    1. Finds the most relevant articles with a positive score (> 1) to the query.
    2. Finds the most relevant articles with a negative score (< 1) to the query.
    3. Finds the most relevant articles with a score between -0.15 and 0.15 to the query.

    Args:
    - query (str): The query string to search for relevant articles.

    Returns:
    - list: A list containing JSON strings for the filtered articles from each search method.
    - [[left_articles_json], [right_articles_json], [middle_articles_json], [all_articles_json]]
    """
    q = QueryChecker(query)
    q.loadData('./temp4.json')

    # [TODO] Exclude duplicate articles based on their titles

    all_indices = q.get_most_similar(query.lower(), articles_df)

    all_matches = articles_df.iloc[all_indices]

    all_matches_filtered = all_matches[[
        'title', 'text', 'score', 'url', 'date']]

    left_matches_filtered = all_matches_filtered[all_matches_filtered['score'] < .05]
    right_matches_filtered = all_matches_filtered[all_matches_filtered['score'] > .05]
    middle_matches_filtered = all_matches_filtered[
        (all_matches_filtered['score'] > -
         0.05) & (all_matches_filtered['score'] < 0.05)
    ]

    left_matches_filtered_json = left_matches_filtered.to_json(
        orient='records')
    right_matches_filtered_json = right_matches_filtered.to_json(
        orient='records')
    middle_matches_filtered_json = middle_matches_filtered.to_json(
        orient='records')
    all_matches_filtered_json = all_matches_filtered.to_json(orient='records')

    return [left_matches_filtered_json, right_matches_filtered_json, middle_matches_filtered_json, all_matches_filtered_json]


@app.route("/")
def home():
    return render_template('base.html', title="sample html")


@app.route("/search")
def test():
    query = request.args.get('q', '')
    q = QueryChecker(query)
    q.loadData('./temp4.json')
    most_similar = q.get_most_similar(query)
    return most_similar


@app.route("/update-score", methods=['POST'])
def feedback():
    data = request.json
    user_score = data["user_score"]
    current_score = data["current_score"]
    title = data["title"]
    # If it is too far away from the original score it will be weighted less
    diff = abs(user_score - current_score)
    weight = 1/(10+(4*diff))
    new_score = (((1-weight)*current_score) + (weight*user_score))
    row_index = articles_df.loc[articles_df["title"] == title].index[0]
    articles_df.loc[row_index, 'score'] = new_score
    with open("temp4.json", "r") as file:
        dataset = json.load(file)
        for item in dataset["articles"]:
            if item["title"] == title:
                item["score"] = new_score

        with open("temp4.json", "w") as file:
            json.dump(dataset, file, indent=4)
    return "done"


@app.route("/articles")
def articles_search():
    text = request.args.get("title")
    # return cos_search(text)]
    # new search has three fields:
    # [[left_articles_json], [right_articles_json], [middle_articles_json], [all_articles_json]]
    # left, right, middle, all = three_search(text)
    return three_searchv2(text)


if 'DB_NAME' not in os.environ:
    app.run(debug=True, host="0.0.0.0", port=5000)
