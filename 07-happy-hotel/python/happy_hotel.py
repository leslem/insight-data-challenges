# # Planning

# ## Challenge

# As a data scientist at a hotel chain, I'm trying to find out what customers are happy and unhappy with, based on reviews. I'd like to know the topics in each review and a score for the topic.

# ## Approach

# - Use standard NLP techniques (tokenization, TF-IDF, etc.) to process the reviews
# - Use LDA to identify topics in the reviews for each hotel
#     - Learn the topics from whole reviews
#     - For each hotel, combine all of the reviews into a metareview
#     - Use the fit LDA model to score the appropriateness of each topic for this hotel
#     - Also across all hotels
# - Look at topics coming up in happy vs. unhappy reviews for each hotel

# ## Results
#

# ## Takeaways
#

# +
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import pyLDAvis  # Has a warning on import
import pyLDAvis.sklearn
import pyLDAvis.gensim
import seaborn as sns

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore, Phrases, TfidfModel  # Has a warning on import
from gensim.parsing.preprocessing import STOPWORDS
from IPython.display import display
from nltk.corpus import stopwords  # Has a warning on import
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from pprint import pprint
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
regex_tokenizer = RegexpTokenizer(r'\w+')
vader_analyzer = SentimentIntensityAnalyzer()
# -

# +
# Plot settings
sns.set(style="whitegrid", font_scale=1.10)
pio.templates.default = "plotly_white"
# -

# +
# Set random number seed for reproducibility
np.random.seed(48)
# -

# +
# Set logging level for gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
# -

# +
data_dir = '~/devel/insight-data-challenges/07-happy-hotel/data'
output_dir = '~/devel/insight-data-challenges/07-happy-hotel/output'
# -

# ## Read in and clean the data

# Before reading in all of the files I downloaded from the GDrive, I used `diff` to compare the files because they looked like they might be duplicates. 
#
# ```
# diff hotel_happy_reviews\ -\ hotel_happy_reviews.csv hotel_happy_reviews\ -\ hotel_happy_reviews.csv.csv
# diff hotel_happy_reviews\ -\ hotel_happy_reviews.csv hotel_happy_reviews(1)\ -\ hotel_happy_reviews.csv
# ```
#
# This indicated that three of the files were exact duplicates, leaving me with one file of happy reviews and one file of not happy reviews.
# ```
# hotel_happy_reviews - hotel_happy_reviews.csv
# hotel_not_happy_reviews - hotel_not_happy_reviews.csv.csv
# ```

# +
happy_reviews = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'hotel_happy_reviews - hotel_happy_reviews.csv'),
)
display(happy_reviews.info())
display(happy_reviews)

# Name this bad_reviews so it's easier to distinguish
bad_reviews = pd.read_csv(
    os.path.join(os.path.expanduser(data_dir), 'hotel_not_happy_reviews - hotel_not_happy_reviews.csv.csv'),
)
display(bad_reviews.info())
display(bad_reviews)
# -

# ### Check that the two dfs are formatted the same

# +
assert happy_reviews.columns.to_list() == bad_reviews.columns.to_list()
assert happy_reviews.dtypes.to_list() == bad_reviews.dtypes.to_list()
# -

# ## Look at the data in detail

# +
display(happy_reviews['hotel_ID'].value_counts())
display(happy_reviews['User_ID'].describe())

display(bad_reviews['hotel_ID'].value_counts())
display(bad_reviews['User_ID'].describe())
# -

# ## Process review text

# ### Tokenize

# Split the reviews up into individual words


# +
def tokenize(review):
    '''Split review string into tokens; remove stop words.

    Returns: list of strings, one for each word in the review
    '''
    s = review.lower()  # Make lowercase
    s = regex_tokenizer.tokenize(s)  # Split into words and remove punctuation.
    s = [t for t in s if not t.isnumeric()]  # Remove numbers but not words containing numbers.
    s = [t for t in s if len(t) > 2]  # Remove 1- and 2-character tokens.
    # I found that the lemmatizer didn't work very well here - it needs a little more tuning to be useful.
    # For example, "was" and "has" were lemmatized to "wa" and "ha", which was counterproductive.
    s = [stemmer.stem(lemmatizer.lemmatize(t, pos='v')) for t in s]  # Stem and lemmatize verbs
    s = [t for t in s if t not in STOPWORDS]  # Remove stop words
    return s


happy_tokens = happy_reviews['Description'].apply(tokenize)
bad_tokens = bad_reviews['Description'].apply(tokenize)

display(happy_tokens.head())
display(bad_tokens.head())

all_tokens = happy_tokens.append(bad_tokens, ignore_index=True)
# -

# ### Find bigrams and trigrams

# Identify word pairs and triplets that are above a given count threshold across all reviews.

# +
# Add bigrams to single tokens
bigrammer = Phrases(all_tokens, min_count=20)
trigrammer = Phrases(bigrammer[all_tokens], min_count=20)

# For bigrams and trigrams meeting the min and threshold, add them to the token lists.
for idx in range(len(all_tokens)):
    all_tokens.iloc[idx].extend([token for token in trigrammer[all_tokens.iloc[idx]]
                                 if '_' in token])  # Bigrams and trigrams are joined by underscores
# -

# ### Remove rare and common tokens, and limit vocabulary

# +
dictionary = Dictionary(all_tokens)
dictionary.filter_extremes(no_below=30, no_above=0.5, keep_n=20000)

# Look at the top 100 and bottom 100 tokens

temp = dictionary[0]  # Initialize the dict

token_counts = pd.DataFrame(np.array(
    [[token_id, dictionary.id2token[token_id], dictionary.cfs[token_id]]
     for token_id in dictionary.keys()
     if token_id in dictionary.cfs.keys() and token_id in dictionary.id2token.keys()
     ]
), columns=['id', 'token', 'count'])

token_counts['count'] = token_counts['count'].astype('int')
token_counts['count'].describe()
token_counts = token_counts.sort_values('count')

plt.rcParams.update({'figure.figsize': (5, 3.5), 'figure.dpi': 200})
token_counts['count'].head(5000).hist(bins=100)
plt.suptitle("Counts for 5,000 least frequent included words")
plt.show()
display(token_counts.head(50))

plt.rcParams.update({'figure.figsize': (5, 3.5), 'figure.dpi': 200})
token_counts['count'].tail(1000).hist(bins=100)
plt.suptitle("Counts for 1,000 most frequent included words")
plt.show()
display(token_counts.tail(50))
# -

# +
# Replace the split data with the data updated with phrases
display(happy_tokens.shape, bad_tokens.shape)
happy_tokens = all_tokens.iloc[:len(happy_tokens)].copy().reset_index(drop=True)
bad_tokens = all_tokens.iloc[len(happy_tokens):].copy().reset_index(drop=True)
display(happy_tokens.shape, bad_tokens.shape)
# -

# ### Look at two examples before and after preprocessing

# +
happy_idx = np.random.randint(1, len(happy_tokens))
bad_idx = np.random.randint(1, len(bad_tokens))

print('HAPPY before:')
display(happy_reviews['Description'].iloc[happy_idx])
print('HAPPY after:')
display(happy_tokens.iloc[happy_idx])

print('NOT HAPPY before:')
display(bad_reviews['Description'].iloc[bad_idx])
print('NOT HAPPY after:')
display(bad_tokens.iloc[bad_idx])
# -

# ### Vectorize with Bag of Words and TF-IDF

# +
bow_corpus = [dictionary.doc2bow(review) for review in all_tokens]
tfidf_model = TfidfModel(bow_corpus)
tfidf_corpus = tfidf_model[bow_corpus]
print('Number of unique tokens: {}'.format(len(dictionary)))
print('Number of documents: {}'.format(len(bow_corpus)))
len(tfidf_corpus)
# -


# ## LDA topic modeling

# +
# Fit a single version of the LDA model.
num_topics = 10
chunksize = 5000
passes = 4
iterations = 200
eval_every = 1  # Evaluate convergence at the end

id2word = dictionary.id2token

lda_model = LdaMulticore(
    corpus=tfidf_corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='symmetric',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every,
    workers=4  # Use all four cores
)

top_topics = lda_model.top_topics(tfidf_corpus)
pprint(top_topics)
# -

# Gensim calculates the [intrinsic coherence score](http://qpleple.com/topic-coherence-to-evaluate-topic-models/) for
# each topic. By averaging across all of the topics in the model you can get an average coherence score. Coherence
# is a measure of the strength of the association between words in a topic cluster. It is supposed to be an objective
# way to evaluate the quailty of the topic clusters. Higher scores are better.

# +
# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)
# -


# References:
# - https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py
# - https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

# +
# This code is used to run the .py script from beginning to end in the python interpreter
# with open('python/happy_hotel.py', 'r') as f:
#     exec(f.read())

# plt.close('all')
