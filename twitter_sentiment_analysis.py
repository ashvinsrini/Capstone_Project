# run from /Users/ashvinsrinivasan/Desktop/Machinelearning/Udacity_dir/Advanced/capstone
import sentiment_pred as sp
import pandas as pd 
import nltk
df=pd.read_csv('/Users/ashvinsrinivasan/Desktop/Machinelearning/Udacity_dir/Advanced/capstone/tweet_sentiment/tweet.csv')
votes=[]
conf=[]
tweet_words=[]
for i in range(len(df)):
    m,n=sp.sentiment(df.loc[i,'tweets'])
    votes.append(m)
    conf.append(n)
    tweet_words.append(nltk.word_tokenize(df.loc[i,'tweets']))
new_df=pd.concat([pd.Series(tweet_words),pd.Series(votes),pd.Series(conf)],axis=1)    
new_df.columns=['tweets','category','confidence']
outputpath='/Users/ashvinsrinivasan/Desktop/Machinelearning/Udacity_dir/Advanced/capstone/tweet_sentiment/final_tweet_analysis.csv'
new_df.to_csv(outputpath)