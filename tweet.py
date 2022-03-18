import re
import pandas as pd
import numpy as np

#Load Dataset Containing Tweets

train = pd.read_csv('./data/train.csv')
print("Training Set:"% train.columns, train.shape, len(train))
test = pd.read_csv('./data/test.csv')
print("Test Set:"% test.columns, test.shape, len(test))

#Data Clining 

combi = train.append(test, ignore_index=True)

def remove_pattern(input_txt, pattern):
  r = re.findall(pattern, input_txt)
  for i in r:
    input_txt = re.sub(i, '', input_txt)

  return input_txt

combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")

# Remove special characters, numbers, punctuation

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " " )

# Removing short words

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

#tokenization
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())
n=len(tokenized_tweet)

profane_tokens = {"nerfherder","spoil","you","they"} #Any Word As Per Your Need

#Empty Array For Storing Profe...
degree_of_profanity = [0] * n
lenth = [0] * n
for i in range(0,n-1):
    lenth[i] = len(tokenized_tweet[i])

print(lenth[0])
for i in range(0,n-1):
    degree_of_profanity[i] = sum(1 for t in tokenized_tweet[i] if t in profane_tokens) / (lenth[i] + 1)



combi['Degree_of_Profanity'] = degree_of_profanity

print(combi.head()) # You Will Have Your Result


