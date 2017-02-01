import logging

from nltk import (bigrams, trigrams, compat, word_tokenize, FreqDist,
                  WordNetLemmatizer)
from nltk.corpus import gutenberg
import numpy as np
from matplotlib import pylab

logging.basicConfig(level=logging.DEBUG)

FILE_ID = 'shakespeare-macbeth.txt'


def main():
    """
    Driver function

    Runs the tasks stated in problem.
    """
    file_id = 'shakespeare-macbeth.txt'
    sent_list = gutenberg.sents(file_id)
    print("Total number of sentences in corpus is : {}".format(len(sent_list)))

    word_list = word_tokenize(gutenberg.raw(file_id))
    print("Total number of words in corpus is {}".format(len(word_list)))
    word_dict = set(word_list)
    print("Total number of unique words in corpus is {}".format(len(word_dict)))

    word_list = [word for word in word_list if len(word) > 1]
    print("Total number of unigrams in corpus is {}".format(len(word_list)))
    uni_freq_dist = FreqDist(word_list)
    table_20_common(uni_freq_dist, label='unigrams')
    plot_most_common(uni_freq_dist, filename='plot1_3.png')

    bi_tokens = list(bigrams(word_list))
    print("Total number of bigrams in corpus is {}".format(len(bi_tokens)))
    bi_freq_dist = FreqDist(bi_tokens)
    table_20_common(bi_freq_dist, label='bigrams')
    plot_most_common(bi_freq_dist, filename='plot1_4.png')

    tri_tokens = list(trigrams(word_list))
    print("Total number of trigrams in corpus is {}".format(len(tri_tokens)))
    tri_freq_dist = FreqDist(tri_tokens)
    table_20_common(tri_freq_dist, label='trigrams')
    plot_most_common(tri_freq_dist, filename='plot1_5.png')

    count_threshold(0.9, uni_freq_dist, "unigrams")
    count_threshold(0.8, bi_freq_dist, "bigrams")
    count_threshold(0.7, tri_freq_dist, "trigrams")

    lemmatized_tokens = lemmatize_tokens(word_list)

    uni_freq_dist = FreqDist(lemmatized_tokens)
    count_threshold(0.9, uni_freq_dist, 'unigrams')

    bi_freq_dist = FreqDist(list(bigrams(lemmatized_tokens)))
    count_threshold(0.8, bi_freq_dist, 'bigrams')

    tri_freq_dist = FreqDist(list(trigrams(lemmatized_tokens)))
    count_threshold(0.7, tri_freq_dist, 'trigrams')


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


def plot_most_common(freq_dist, max_x=20, filename='plot.png'):
    """
    Save a line graph of frequency of most common
    n-grams in frequency distribution. By default,
    20 most common tokens are plotted.
    :param freq_dist: FreqDist instance
    :param max_x: Number of most common tokens to plot
    :param filename: Name of image file
    :return: None
    """
    items = [item for item, _ in freq_dist.most_common(max_x)]
    freq = list(freq_dist._cumulative_frequencies(items))
    pylab.grid(True, color='silver')
    pylab.plot(freq)
    pylab.xticks(range(len(items)), [compat.text_type(s) for s in items], rotation=90)
    fig = pylab.gcf()
    fig.savefig(filename, dpi=fig.dpi)
    pylab.close()


def count_threshold(threshold, freq_dist, label):
    """
    Count number of n-gram tokens that make threshold
    fraction of corpus.
    :param threshold: Fraction
    :param freq_dist: FreqDist instance
    :param label: degree of n-grams
    :return: None
    """
    arr = np.array(list(reversed(sorted([val for _, val in freq_dist.items()]))))
    total_grams = np.sum(arr)
    threshold_grams = np.argmin(arr.cumsum() < total_grams*threshold)
    print("{} {} are required to cover {} % of total corpus.".format(
        threshold_grams, label, threshold*100
    ))


def lemmatize_tokens(word_list):
    """
    Lemmatize a list of tokens using WordNetLemmatizer.
    :param word_list: list of tokens
    :return: lemmatized list of tokens
    """
    wnl = WordNetLemmatizer()
    lemmatized_list = [wnl.lemmatize(token) for token in word_list]
    return lemmatized_list

if __name__ == "__main__":
    main()
