# %%
"""
# NLTK provides a number of functions out of the box that you can call with few or no arguments
that will help you meaningfully analyze text before you even touch its machine learning capabilities.

Many of NLTK’s utilities are helpful in preparing your data for more advanced analysis.
First: some tools out-of-the-box:

- word tokenizers & stop words
- frequence analisys
- Concordance and collocations: words & its surroundings

"""
import nltk
import pandas as pd


# %%
"""
Check the word_tokenizer from NLTK:
function for get the list of tokens (words + signs) from a text passed
"""

TEXT = """
For some quick analysis, creating a corpus could be overkill.
If all you need is a word list,
there are simpler ways to achieve that goal. total overkill. Overkill, I tell you. 123 12"""

tkn_words: list[str] = [w.lower() for w in nltk.word_tokenize(TEXT) if w.isalpha()]
set(tkn_words)

# %%
"""
To know how to work with a corpus, lets load one and turn it toa list of words
"""
# Working with a corpus: loading the State of the Union corpus in a list of words
words: list[str] = [w.lower() for w in nltk.corpus.state_union.words() if w.isalpha()]
# for help with corpus functions help(nltk.corpus.state_union)
words

# %%
"""
stop_words:  “of,” “a,” “the,” and similar.
 These common words are called stop words, and they can have a negative effect on analysis because ther frequency.
 We can filter them out.
"""
# remove stopwords

stopwords: list[str] = nltk.corpus.stopwords.words("english")
words: list[str] = [w for w in words if w.lower() not in stopwords]
words
# %%
"""
lets do this too with the tkn_words we obtained earlier so we work with a smaller list
"""
tkn_words: list[str] = [w for w in tkn_words if w.lower() not in stopwords]
tkn_words


# %%
"""
1. Creating Frequency Distributions
A frequency distribution is essentially a table that tells you how many times each word appears
within a given text. Basic Operation

In NLTK, frequency distributions are a specific object type (nltk.probability.FreqDist)
This class provides useful operations for word frequency analysis.

To build a frequency distribution with NLTK, construct the FreqDist class with any word list
"""
# Frequence distribution

tkn_fd: nltk.probability.FreqDist = nltk.FreqDist(tkn_words)
pd.DataFrame(tkn_fd, index=[0])

# %%
# With the complete corpus word list
# first passed to lowercase
words = [w.lower() for w in words]
fd: nltk.probability.FreqDist = nltk.FreqDist(words)
# fd: nltk.probability.FreqDist = words.vocab()  Equivalent
pd.DataFrame(fd, index=[0])


# %%
# After building the object, you can use methods like .most_common() and .tabulate()
# to to quickly determine frequently used words in a sample:

fd.most_common(3), fd.tabulate(5)

# %%
# or to locate the words most used wich contains certain letters.
# You could create frequency distributions of words starting with a particular letter,
# or of a particular length, or containing certain letters.

fd_ent_words: nltk.probability.FreqDist = nltk.FreqDist([w for w in words if "ry" in w])
fd_ent_words.tabulate(5)

# %%
"""2. Concordance and collocations, the words and its sourroundings.

## Concordance
In the context of NLP, a concordance is a collection of word locations along with their context. 
You can use concordances to find:
- How many times a word appears
- Where each occurrence appears
- What words surround each occurrence
In NLTK, you can do this by calling .concordance(). Note that it ignores case

To use it, you need an instance of the nltk.Text class, which can also be constructed with a word list.
"""
# for this we need the original corpus, with stopwords and so on so the sentences actually make sense
text = nltk.Text(nltk.corpus.state_union.words())
# directly to the console
text.concordance("america", lines=5)

# %%
# can be obtained in list format
concordance_list = text.concordance_list("america", lines=2)
for entry in concordance_list:
    print(entry.line)

# %%
"""Concordance and collocations, the words and its sourroundings.

## Collocation 
Words that come together in Bigrams, trigrams & quadrams:
Bigrams: Frequent two-word combinations
Trigrams: Frequent three-word combinations
Quadgrams: Frequent four-word combinations
"""

finder3 = nltk.collocations.TrigramCollocationFinder.from_words(words)
finder3.ngram_fd.most_common(2), finder3.ngram_fd.tabulate(5)
# %%
finder2 = nltk.collocations.BigramCollocationFinder.from_words(words)
finder2.ngram_fd.tabulate(7)

# %%
"""
Using NLTK’s Pre-Trained Sentiment Analyzer (ENGLISH)

NLTK already has a built-in, pretrained sentiment analyzer called 
VADER (Valence Aware Dictionary and sEntiment Reasoner).
"""
# analisys with included sentiment analist VADER (only english)
from nltk.sentiment import SentimentIntensityAnalyzer


sia = SentimentIntensityAnalyzer()
sia.polarity_scores("Wow, NLTK is really powerful!")
# %%
# analisys with multilanguage sentiment analist vader-multi
# but there is quota
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIAmulti

siam = SIAmulti()
siam.polarity_scores(
    "Pero el que de verdad es bueno es el siamulti"
), siam.polarity_scores(
    "Esta libreria, siamulti, es realmente poderosa!"
), siam.polarity_scores(
    "vader es una mierda"
), siam.polarity_scores(
    "✈️En dos dias estaré de vuelta en la lluviosa Asturias :-(☔️"
)

# %%
#  list of raw tweets as strings isntead word by word.
tweets = [t.replace("://", "//") for t in nltk.corpus.twitter_samples.strings()]
# for help with corpus functions help(nltk.corpus.twitter_samples)

# %%
from random import shuffle


def is_positive_c_sent(tweet: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(tweet)["compound"] > 0


shuffle(tweets)
for tweet in tweets[:10]:
    print(">", is_positive_c_sent(tweet), tweet)
# %%
positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])
all_review_ids = positive_review_ids + negative_review_ids

# %%
## Measure sentiment (compound score) with movies  corpus
from statistics import mean


def is_positive(review_id: str) -> bool:
    """True if the average of all sentence compound scores is positive."""
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(text)
    ]
    return mean(scores) > 0


# %%


def is_positivec(review_id: str) -> bool:
    """True if the count of all sentence with positive compound scores
    is higher than the negative."""
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores_pos = [
        1
        for sentence in nltk.sent_tokenize(text)
        if sia.polarity_scores(sentence)["compound"] > 0
    ]
    scores_neg = [
        1
        for sentence in nltk.sent_tokenize(text)
        if sia.polarity_scores(sentence)["compound"] < 0
    ]

    return len(scores_pos) - len(scores_neg) > 0


# %%
shuffle(all_review_ids)
correct = 0
for review_id in all_review_ids:
    if is_positive(review_id):
        if review_id in positive_review_ids:
            correct += 1
    else:
        if review_id in negative_review_ids:
            correct += 1

print(f"{correct / len(all_review_ids):.2%} correct")
# %%
shuffle(all_review_ids)
correct = 0
for review_id in all_review_ids:
    if is_positivec(review_id):
        if review_id in positive_review_ids:
            correct += 1
    else:
        if review_id in negative_review_ids:
            correct += 1

print(f"{correct / len(all_review_ids):.2%} correct")
# %%
