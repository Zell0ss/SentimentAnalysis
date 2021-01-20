# %%
"""
# NLTK provides a number of functions out of the box that you can call with few or no arguments
that will help you meaningfully analyze text before you even touch its machine learning capabilities.

Many of NLTK’s utilities are helpful in preparing your data for more advanced analysis.

"""
import nltk
import pandas as pd


# %%
# Check the tokenizer from NLTK
TEXT = """
For some quick analysis, creating a corpus could be overkill.
If all you need is a word list,
there are simpler ways to achieve that goal. total overkill. Overkill, I tell you."""

tkn_words: list[str] = [w.lower() for w in nltk.word_tokenize(TEXT) if w.isalpha()]
tkn_words

# %%
# Working with a corpus: loading the State of the Union corpus in a list of words
words: list[str] = [w.lower() for w in nltk.corpus.state_union.words() if w.isalpha()]
# for help with corpus functions help(nltk.corpus.state_union)
words

# %%
# remove stopwords

stopwords: list[str] = nltk.corpus.stopwords.words("english")
words: list[str] = [w for w in words if w.lower() not in stopwords]
words
# %%
tkn_words: list[str] = [w for w in tkn_words if w.lower() not in stopwords]
tkn_words

# %%
# Frequence distribution

tkn_fd: nltk.probability.FreqDist = nltk.FreqDist(tkn_words)
pd.DataFrame(tkn_fd, index=[0])

# %%
# After building the object, you can use methods like
# .most_common() and .tabulate() to start visualizing information:

fd: nltk.probability.FreqDist = nltk.FreqDist(words)
# fd: nltk.probability.FreqDist = words.vocab()  Equivalent

fd.most_common(3), fd.tabulate(5)  # pylint: disable=expression-not-assigned

# %%
ent_words: nltk.probability.FreqDist = nltk.FreqDist([w for w in words if "ent" in w])
ent_words.tabulate(10)
# %%
# Concordance, the words and its sourroundings.
# for this we need the original corpus, with stopwords and so on

text = nltk.Text(nltk.corpus.state_union.words())
text.concordance("america", lines=5)
# %%
# Collocation words that come together in Bigrams, trigrams & quadrams
finder3 = nltk.collocations.TrigramCollocationFinder.from_words(words)
finder3.ngram_fd.most_common(2), finder3.ngram_fd.tabulate(5)
# %%
finder2 = nltk.collocations.BigramCollocationFinder.from_words(words)
finder2.ngram_fd.tabulate(7)
# %%
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
), siam.polarity_scores("siamulti es realmente poderoso!"), siam.polarity_scores(
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


def is_positive(tweet: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(tweet)["compound"] > 0


shuffle(tweets)
for tweet in tweets[:10]:
    print(">", is_positive(tweet), tweet)
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
    """True if the countof all sentence with positive compound scores is higher than the negative."""
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
