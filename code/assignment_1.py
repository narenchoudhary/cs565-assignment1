from nltk import (bigrams, trigrams, compat, word_tokenize, FreqDist)
from nltk.corpus import gutenberg
from matplotlib import pylab

FILE_ID = 'shakespeare-macbeth.txt'


def main():
    file_id = 'shakespeare-macbeth.txt'
    sent_list = gutenberg.sents(file_id)
    print("Total number of sentences in corpus is : {}".format(len(sent_list)))

    word_list = word_tokenize(gutenberg.raw(file_id))
    print("Total number of words in corpus is {}".format(len(word_list)))
    word_dict = set(word_list)
    print("Total number of unique words in corpus is {}".format(len(word_dict)))

    word_list = [word for word in word_list if len(word) > 1]
    print("Total number of unigrams in corpus is {}".format(len(word_list)))
    freq_dist = FreqDist(word_list)
    table_20_common(freq_dist, label='unigrams')
    plot_20_common(freq_dist, filename='plot1_3.png')

    bi_tokens = list(bigrams(word_list))
    print("Total number of bigrams in corpus is {}".format(len(bi_tokens)))
    freq_dist = FreqDist(bi_tokens)
    table_20_common(freq_dist, label='bigrams')
    plot_20_common(freq_dist, filename='plot1_4.png')

    tri_tokens = list(trigrams(word_list))
    print("Total number of trigrams in corpus is {}".format(len(tri_tokens)))
    freq_dist = FreqDist(tri_tokens)
    table_20_common(freq_dist, label='trigrams')
    plot_20_common(freq_dist, filename='plot1_5.png')


def table_20_common(freq_dist, label=''):
    freq_dist_20 = freq_dist.most_common(20)
    print("20 most common {} in the frequency distribution are:".format(label))
    print('| {} | count |'.format(label))
    print('|:----|:----:|')
    for item, item_count in freq_dist_20:
        print("| {} \t | {}".format(item, item_count))


def plot_20_common(freq_dist, max_x=20, filename='plot.png'):
    items = [item for item, _ in freq_dist.most_common(max_x)]
    freq = list(freq_dist._cumulative_frequencies(items))
    pylab.grid(True, color='silver')
    pylab.plot(freq)
    pylab.xticks(range(len(items)), [compat.text_type(s) for s in items], rotation=90)
    fig = pylab.gcf()
    fig.savefig(filename, dpi=fig.dpi)
    pylab.close()


if __name__ == "__main__":
    main()
