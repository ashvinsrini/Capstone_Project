import tweepy
import os
import pandas as pd
#### All the keys are got by signing into twitter app. You can create them by signing into your twitter account.
consumer_key='Insert your consumer key here'
consumer_secret='Insert your consumer secret here'
access_token='Insert your access token here'
access_token_secret='Insert your access token secret here'

auth=tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)


api=tweepy.API(auth)

tweets=api.search('cricket')
tweet_list=[]
for tweet in tweets:
    tweet_list.append(tweet.text)
    print(tweet.text)

path = os.getcwd()
new_dir='tweet_sentiment'
dir_path=path +'/' + new_dir
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
outputpath=dir_path+'/'+'tweet.csv'

df_tweet_list=pd.DataFrame(tweet_list)
df_tweet_list.columns=['tweets']
df_tweet_list.to_csv(outputpath)