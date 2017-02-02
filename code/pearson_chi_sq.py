from __future__ import division

from nltk import ngrams, FreqDist


class PearsonChiSqTest(object):

    def __init__(self, word_list):
        """
        Initialize the PearsonChiSqTest instance with word_list.
        :param word_list: List of tokens
        """
        self.word_list = word_list
        self.uni_fd = FreqDist(word_list)
        self.bi_fd = FreqDist(list(ngrams(word_list, 2)))
        self.collocation_scores = []

    def filter_frequency(self, low=5):
        """
        Ignore bi-grams whose frequency is below ``low``.
        :param low: minimum allowed frequency
        :return: None
        """
        new_fd = FreqDist()
        for key, value in self.bi_fd.items():
            if value >= low:
                new_fd[key] = value
        self.bi_fd = new_fd

    def calc_scores(self):
        """
        Calculate chi-square scores each bi-gram.
        :return: None
        """
        for sample in self.bi_fd.keys():
            mat_11, mat_12, mat_21, mat_22 = self.contingency_matrix(sample)

            numerator = (mat_11*mat_22 - mat_12*mat_21)**2
            denominator = (mat_11 + mat_21)*(mat_11 + mat_12)*(
                mat_12 + mat_22)*(mat_21 + mat_22)
            score = (len(self.word_list)*numerator)/denominator
            self.collocation_scores.append((sample, round(score, 4)))

    def sort_scores(self):
        """
        Sort chi-square scores.
        :return: None
        """
        self.collocation_scores = sorted(
            self.collocation_scores, key=lambda tup: tup[1])

    def top_collocations(self, n=20):
        """
        Return top ``n`` collocations with highest score in descending order.
        :param n: Number of collocations required
        :return: None
        """
        return list(reversed([collocation for collocation, _ in self.collocation_scores[-n:]]))

    def contingency_matrix(self, sample):
        """
        Contingency matrix of tokens of a bi-gram.
        :param sample: Tuple of 2 tokens
        :return:
        """
        num_all_all = len(self.word_list)
        num_w1_w2 = self.bi_fd[sample]
        num_w1_all = self.uni_fd[sample[0]]
        num_all_w2 = self.uni_fd[sample[1]]
        num_w1_not_w2 = num_w1_all - num_w1_w2
        num_not_w1_w2 = num_all_w2 - num_w1_w2
        num_not_w1_not_w2 = (num_all_all - num_w1_w2 - num_not_w1_w2
                             - num_w1_not_w2)
        return num_w1_w2, num_not_w1_w2, num_w1_not_w2, num_not_w1_not_w2
