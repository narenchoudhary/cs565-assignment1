import math
import logging

from matplotlib import pylab, pyplot
from nltk import (compat, word_tokenize, FreqDist, WordNetLemmatizer, ngrams)
from nltk.corpus import gutenberg
import numpy as np

from pearson_chi_sq import PearsonChiSqTest

logging.basicConfig(level=logging.DEBUG)

FILE_ID = 'shakespeare-macbeth.txt'


def analyse_sents():
    """
    Analyze sentences present in corpus.
    :return:
    """
    sent_list = gutenberg.sents(FILE_ID)
    print("Total number of sentences in corpus is {}".format(len(sent_list)))


def analyze_words(raw_text):
    """
    Analyze words present in corpus.
    :param raw_text: raw text
    :type raw_text: str
    :return:
    """
    word_list = word_tokenize(raw_text)
    print("Total number of words in corpus is {}".format(len(word_list)))
    word_dict = set(word_list)
    print("Total number of unique words in corpus is {}".format(len(word_dict)))


def get_tokens(raw_text, method='word_tokenize', len_filter=0):
    """
    Get tokens from raw text using a specific method.
    :param raw_text: raw text
    :type raw_text: str
    :param method: method to be used
    :type method: str
    :param len_filter: filter tokens of lengths below len_filter
    :type len_filter: int
    :return: list of tokens
    """
    token_list = None
    if method == 'word_tokenize':
        token_list = word_tokenize(raw_text)
    if len_filter > 0:
        token_list = [token for token in token_list if len(token) > len_filter]
    return token_list


def get_label(n=1, plural=False):
    """
    Ger label for n-gram.
    :param n: the degree of the n-grams
    :type n: int
    :param plural: return plural form of label if True
    :type plural: bool
    :return: str
    """
    if n == 1:
        return "unigram" if plural is False else "unigrams"
    if n == 2:
        return "bigram" if plural is False else "bigrams"
    if n == 3:
        return "trigram" if plural is False else "trigrams"


def freq_plot(len_list, n):
    freq_dist = FreqDist(len_list)
    frequencies = [freq_dist[sample] for sample in freq_dist.keys()]
    pos = np.arange(len(freq_dist.keys()))
    width = 1.0
    ngram_label = get_label(n, True)
    ax = pyplot.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(pos)
    ax.set_xlabel(ngram_label + " length")
    ax.set_ylabel(ngram_label + " count")
    ax.set_title("Frequency plot of {} by length".format(ngram_label))
    ax.grid(True)
    pyplot.ylim(0, math.ceil(max(frequencies)/500)*500)
    pyplot.bar(pos, frequencies, width, color='b', edgecolor='k')
    figure = pyplot.gcf()
    figure.savefig(ngram_label, dpi=figure.dpi)
    pyplot.close()


def analyze_ngrams(token_list, n=1, threshold=1.0, lemmatize=False, **kwargs):
    """
    Analyze n-grams.
    :param token_list: list of tokens to generate n-grams from
    :type token_list: list
    :param n: the degree of n-grams (n=2 for bigrams)
    :type n: int
    :param threshold: threshold fraction
    :type threshold: float
    :param lemmatize: If True, lemmatize tokens before analysis
    :type lemmatize: bool
    :param kwargs: keyword arguments (Eg. report)
    :type kwargs: dict
    :return: None
    """
    if lemmatize:
        wnl = WordNetLemmatizer()
        token_list = [wnl.lemmatize(token) for token in token_list]
    ngrams_list = list(ngrams(token_list, n))
    freq_dist = FreqDist(ngrams_list)
    freq_list = np.array(list(reversed(sorted([val for _, val in freq_dist.items()]))))
    threshold_count = np.argmin(freq_list.cumsum() < freq_dist.N()*threshold)

    report = kwargs['report'] if 'report' in kwargs else False
    if report:
        print("Total number of {} in corpus is {}".format(
            get_label(n, True), freq_dist.N()
        ))
        print("Total number of unique {} in corpus is {}".format(
            get_label(n, True), freq_dist.B()
        ))
        print("{} {} are required to cover {} % of total corpus.".format(
            threshold_count, get_label(n, True), threshold * 100
        ))

    gen_plot = kwargs['plot'] if 'plot' in kwargs else False
    if gen_plot:
        if n == 3:
            len_list = [len(n_tuple[0]) + len(n_tuple[1]) + len(n_tuple[2])
                        for n_tuple in ngrams_list]
        elif n == 2:
            len_list = [len(n_tuple[0]) + len(n_tuple[1]) for n_tuple in
                        ngrams_list]
        else:
            len_list = [len(n_tuple[0]) for n_tuple in ngrams_list]
        freq_plot(len_list, n)


def table_20_common(freq_dist, label=''):
    """
    Display a markdown formatted table for 20 most common
    n-grams in frequency distribution.
    :param freq_dist: FreqDist instance
    :param label: degree of n-gram
    :return: None
    """
    freq_dist_20 = freq_dist.most_common(20)
    print("20 most common {} in the frequency distribution are:".format(label))
    print('| {} | count |'.format(label))
    print('|:----|:----:|')
    for item, item_count in freq_dist_20:
        print("| {} \t | {}".format(item, item_count))


def part_1_2():
    """
    Driver function for part 1 and part 2 of assignments.s
    :return:
    """
    raw_text = gutenberg.raw(FILE_ID)

    # tasks related to sentence
    analyse_sents()

    # tasks related to words
    analyze_words(raw_text)

    # get list of tokens of length > 1
    token_list = get_tokens(raw_text, len_filter=1)

    print("\nAnalysis of tokens before lemmatization:")
    analyze_ngrams(token_list, n=1, threshold=0.9, report=True, plot=True)
    analyze_ngrams(token_list, n=2, threshold=0.8, report=True, plot=True)
    analyze_ngrams(token_list, n=3, threshold=0.7, report=True, plot=True)

    print("\nAnalysis of tokens after lemmatization:")
    analyze_ngrams(token_list, n=1, threshold=0.9, lemmatize=True, report=True)
    analyze_ngrams(token_list, n=2, threshold=0.8, lemmatize=True, report=True)
    analyze_ngrams(token_list, n=3, threshold=0.7, lemmatize=True, report=True)


def test_pearson():
    """
    For all bi-grams, test if the bi-gram is a valid collocation candidate
    using pearson's chi-square test and print best collocation candidates 4
    based on chi-square score.
    :return: None
    """
    word_list = word_tokenize(gutenberg.raw(FILE_ID))
    word_list = [word for word in word_list if len(word) > 1]

    pearson_chi = PearsonChiSqTest(word_list)
    pearson_chi.filter_frequency(low=5)
    pearson_chi.calc_scores()
    pearson_chi.sort_scores()
    top_cols = pearson_chi.top_collocations()
    for col in top_cols:
        print("{} {}  ".format(col[0], col[1]))

if __name__ == "__main__":
    part_1_2()
