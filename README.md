# SentimentAnalysis
https://realpython.com/python-nltk-sentiment-analysis/


## What is Sentiment Analisis

Sentiment analysis is the practice of using algorithms to classify various samples of related text into overall positive and negative categories. Sentiment analysis can help you determine the ratio of positive to negative engagements about a specific topic. 

We will use NLTK for processing the text data and perform sentiment analysis.

## NLTK library

The NLTK library contains various utilities that allow you to effectively manipulate and analyze linguistic data. We will focus in its text classifiers using them for sentiment analysis.

**corpus and corpora**:  Pre-processed bodies of text used to train the models. A corpus is a large collection of related text samples, but inthe context of NLTK, means compiled with features for natural language processing (NLP), such as categories and numerical scores for particular features.

### Installing NLTK

You need not only to install NLTK but use its download manager to get its resources

```` shell
$ .venv/bin/activate
$ pip install nltk
$ python
>>> import nltk
>>> nltk.download()
````

NLTK will display its downloader and the resources in o https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml

For spanish we have cess_esp, inaugural, machado, opinion_lexicon & spanish_grammars

For this tutorial we need installed:
- names: A [list of common English names](http://www-2.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/) compiled by Mark Kantrowitz
- stopwords: A list of really common words, like articles, pronouns, prepositions, and conjunctions
- state_union: A sample of transcribed [State of the Union](https://en.wikipedia.org/wiki/State_of_the_Union) addresses by different US presidents, compiled by Kathleen Ahrens
- twitter_samples: A list of social media phrases posted to Twitter
- movie_reviews: [Two thousand movie reviews](http://www.cs.cornell.edu/people/pabo/movie-review-data/) categorized by Bo Pang and Lillian Lee
- averaged_perceptron_tagger: A data model that NLTK uses to categorize words into their part of speech
- vader_lexicon: A scored [list of words and jargon](https://github.com/cjhutto/vaderSentiment), specifically attuned to sentiments expressed in social media, that NLTK references when performing sentiment analysis, created by C.J. Hutto and Eric Gilbert
- punkt: A data model created by Jan Strunk that NLTK uses to split full texts into word lists

For this you can run the script `1.prepare_nltk.py`

```` shell
>>> import nltk

>>> nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])
````
Once ready go to the `2.compilig_data.py`