import math

import textacy
from matplotlib import pyplot
from nltk.corpus import gutenberg
import numpy as np

from assignment_1 import get_label

FILE_ID = 'shakespeare-macbeth.txt'


def analyse_sents(doc):
    """
    Analyze sentences present in corpus.
    :param doc: Doc object
    :type doc: Doc
    :return: None
    """
    print("Total number of sentences in corpus is {}".format(doc.n_sents))


def analyse_words(doc):
    """
    Analyze words present in corpus.
    :param doc: Doc object
    :type doc: Doc
    :return: None
    """
    word_list = list(textacy.extract.words(doc, filter_stops=False, 
            filter_punct=False))
    print(type(word_list))
    print(word_list[0:50])
    print("Total number of words present in corpus is {}".format(
        len(word_list)))
    print("Total number of words unique present in corpus is {}".format(
        len(list(doc.to_bag_of_words()))))


def get_freq_dist(ngrams, n):
    """
    Return frequency distribution of token length in ngrams provided.
    :param ngrams: list of ngrams
    :type ngrams: list
    :param n: degree of ngrams
    :type n: int
    :return: numpy array
    """
    if n == 3:
        len_list = [len(wl[0]) + len(wl[1]) + len(wl[2]) for wl in ngrams]
    elif n == 2:
        len_list = [len(wl[0]) + len(wl[1]) for wl in ngrams]
    else:
        len_list = [len(wl[0]) for wl in ngrams]
    unique, counts = np.unique(len_list, return_counts=True)
    freq_dist = np.asarray((unique, counts)).T
    return freq_dist


def freq_plot(ngrams, n):
    print("Generating figure:")
    freq_dist = get_freq_dist(ngrams, n)
    print(type(freq_dist))
    print(freq_dist)
    frequencies = [i[1] for i in freq_dist]
    pos = np.arange(len(frequencies))
    width = 1.0
    ngram_label = get_label(n, True)
    print("label is : {}".format(ngram_label))
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
    file_name = "spacy_{}".format(ngram_label)
    figure.savefig(file_name, dpi=figure.dpi)
    print("figure saved")
    pyplot.close()


def analyze_ngrams(doc, n):
    """
    Perform tasks asked in Part #1 of assignment.
    :param text_blob: TextBlob instance
    :type text_blob: TextBlob
    :param n: degree of ngram
    :type n: int
    :return: None
    """
    ngrams = list(textacy.extract.ngrams(doc, n, filter_stops=False, 
            filter_punct=False))
    label = get_label(n, True)
    print("Total number of {} in corpus is {}.".format(label, len(ngrams)))
    #freq_plot(ngrams, n)


def main():
    raw_text = gutenberg.raw(FILE_ID)
    doc = textacy.doc.Doc(raw_text)
    analyse_sents(doc)
    analyse_words(doc)
    analyze_ngrams(doc, 1)
    analyze_ngrams(doc, 2)
    analyze_ngrams(doc, 3)


if __name__ == "__main__":
    main()
