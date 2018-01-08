# Dependencies
import tweepy
import json
import pandas as pd
import numpy as np

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
consumer_key = "onU0Q8ycGli3zQ4odIvXqxsv6"
consumer_secret = "6RJ27SZQycSC1cTXySoYD3dvWTDz1k28rlgY47nDTrFsCD0MCw"
access_token = "948782946869166085-vwBcdh9wspiLQkDaQaeQ4oZSLxBMZ48"
access_token_secret = "xA1cAcAZhAGOmREWV3qnsyAkyBBaKZvvAZ5yt4uETp9ZD"

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# User Twitter Accounts
users = ["@BBC", "@CNN", "@FOXNEWS", "@NYTIMES","@CBS"]

# Variables for holding sentiments
sources = []
texts = []
dates = []
compound_list = []
positive_list = []
negative_list = []
neutral_list = []


for user in users :
    
    # Loop through 5 pages of tweets (total 100 tweets)
    for x in range(5):
        # Get all tweets from home feed
        public_tweets = api.user_timeline(user, page=x)

        # Loop through all tweets
        for tweet in public_tweets:

          # Run Vader Analysis on each tweet
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
            
            dates.append(tweet["created_at"])
            sources.append(user)
            texts.append(tweet["text"])
            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)

public_tweets = api.user_timeline(user, page=5)
tweet = public_tweets[0]
compound = analyzer.polarity_scores(tweet["text"])["compound"]
pos = analyzer.polarity_scores(tweet["text"])["pos"]
neu = analyzer.polarity_scores(tweet["text"])["neu"]
neg = analyzer.polarity_scores(tweet["text"])["neg"]


# Add each value to the appropriate list
dates.append(tweet["created_at"])
sources.append("@CBS")
texts.append(tweet["text"])
compound_list.append(compound)
positive_list.append(pos)
negative_list.append(neg)
neutral_list.append(neu)
            
            
# Print the Averages
print("")
print("User: %s" % user)
print(f"Compound: {np.mean(compound_list)}")
print(f"Positive: {np.mean(positive_list)}")
print(f"Neutral: {np.mean(neutral_list)}")
print(f"Negative: {np.mean(negative_list)}")
 
print(compound_list)


# print(json.dumps(tweet, sort_keys=True, indent=4, separators=(',', ': ')))

newstweets = pd.DataFrame({"text": texts, "compound": compound_list, "neutral": neutral_list, "positive": positive_list, "negative": negative_list, "user" : sources, "date" : dates})
newstweets.to_csv("newstweets.csv")
newstweets

count = 0
for source in sources:
    if source == "@CBS" :
        count +=1
print(count)

users = ["@BBC", "@CNN", "@FOXNEWS", "@NYTIMES","@CBS"]
newsgroup = newstweets.groupby("user")
newscomparison = newsgroup.mean()
newscomparison

# Scatterplot of 100 tweets of each media outlet
import matplotlib.pyplot as plt

plt.scatter(range(100), newstweets[newstweets['user'] == '@BBC']['compound'], c='pink', alpha=0.9, linewidth=.5, label='BBC')
plt.scatter(range(100), newstweets[newstweets['user'] == '@CNN']['compound'], c='blue', alpha=0.9, linewidth=.5, label='CNN')
plt.scatter(range(100), newstweets[newstweets['user'] == '@CBS']['compound'], c='red', alpha=0.9, linewidth=.5, label='CBS')
plt.scatter(range(100), newstweets[newstweets['user'] == '@FOXNEWS']['compound'], c='green', alpha=0.9, linewidth=.5, label='FOXNEWS')
plt.scatter(range(100), newstweets[newstweets['user'] == '@NYTIMES']['compound'], c='yellow', alpha=0.9, linewidth=.5, label='NYTIMES')
plt.title("Sentiment Analysis of Media Tweets (1/7/2018)")
plt.xlabel('Tweets Ago')
plt.ylabel('Tweet Polarity')
plt.legend(title="Media Outlet", loc='upper center', bbox_to_anchor=(1.2, 0.5))
plt.savefig('sentimentanalysis.png')
plt.show()

# Overall Media Sentiment Bar Graph
mediaoutlets = ["BBC", "CNN", "CBS", "Fox News", "NY Times"]
tweetpolarity = [0.102894, 0.343236, -0.031030, -0.100285, -0.043024]
x_axis = np.arange(len(tweetpolarity))

# Create a bar chart based upon the above data
plt.bar(x_axis, tweetpolarity, color="b", align="edge")
tick_locations = [value+0.4 for value in x_axis]
plt.xticks(tick_locations, mediaoutlets)

# Give the chart a title, x label, and y label
plt.title("Overall Media Sentiment Based on Twitter (1/7/2018)")
plt.xlabel("Media Outlet")
plt.ylabel("Tweet Polarity")

# Save an image of the chart and print it to the screen
plt.savefig("overallsentiment.png")
plt.show()