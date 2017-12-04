# coding=utf-8
import numpy as np
from itertools import combinations_with_replacement
from my_log import logging

training_data_file_name = "assets/mini_training_data.txt"
test_data_file_name = "assets/mini_test_data.txt"
output_file_name = "assets/result.txt"


def load_transcription(transcription_file_name):
    """

    :return: a list of tuple:
        [
        (word: string, phones: list),
        (word: string, phones: list),
        ...,
        (word: string, phones: list),
        ]
    """
    transcription_list = list()
    with open(transcription_file_name, "r") as training_word_file:
        while 1:
            lines = training_word_file.readlines(10000)
            if not lines:
                break
            for line in lines:
                line = line.strip()
                word = line.split("\t")[0]
                phones = line.split("\t")[1].split(" ")
                transcription_list.append((word, phones))
                pass
        pass
        transcription_list = transcription_list
        logging.debug("transcription_list:")
        logging.debug(transcription_list)
        return transcription_list


def load_grapheme_dict(transcription_list):
    """

    :return: a dictionary of grapheme-id pair like: {"a": 0, "b": 1, "c": 2, ...,}
    """
    grapheme_set = set()
    for (word, _) in transcription_list:
        grapheme_set = grapheme_set.union(word)
        pass
    grapheme_list = list(grapheme_set)
    grapheme_dict = dict()
    for i in range(len(grapheme_list)):
        grapheme_dict[grapheme_list[i]] = i
        pass
    grapheme_dict = grapheme_dict
    logging.debug("grapheme_dict:")
    logging.debug(grapheme_dict)
    return grapheme_dict


def load_phoneme_dict(transcription_list):
    """

    :return: a dictionary of phoneme-id pair like: {"ey1":0, "b":1, "iy2": 2, "s": 3, "iy2": 4, ...,}
    """
    phoneme_set = set()
    for (_, phones) in transcription_list:
        phoneme_set = phoneme_set.union(phones)
        pass
    phoneme_list = list(phoneme_set)
    phoneme_list.append("*")
    phoneme_dict = dict()
    for i in range(len(phoneme_list)):
        phoneme_dict[phoneme_list[i]] = i
        pass
    phoneme_dict = phoneme_dict
    logging.debug("phoneme_dict:")
    logging.debug(phoneme_dict)
    return phoneme_dict


def introduce_epsilon_phone_seq(word, phones):
    """

    :param word:
    :param phones:
    :return: a list containing all word-phones pairs with epsilon introduced
    """
    length_diff = len(word) - len(phones)
    if length_diff < 0:
        logging.error("Word length is less than phones'!")
        logging.info(word + "-" + str(phones))
    location_combines_with_replace = [c for c in combinations_with_replacement(range(len(phones) + 1), length_diff)]
    pair_list = list()
    for locations in location_combines_with_replace:
        temp_phones = phones.copy()
        for i in range(len(locations)):
            temp_phones.insert(locations[i] + i, "*")
            pass
        pair_list.append((word, temp_phones))
        pass
    return pair_list


def is_prob_matrix_equal(last_prob_matrix, new_prob_matrix, epsilon):
    """

    :param last_prob_matrix: numpy array.
    :param new_prob_matrix: numpy array.
    :param epsilon:
    :return: True: if mean-square error <= epsilon
                False: if mean-square error > epsilon
    """
    diff_mean = np.mean(np.subtract(last_prob_matrix, new_prob_matrix))
    if diff_mean <= epsilon:
        return True
    return False


def path_to_string(path_list):
    """

    :param path_list: a list of dtw path result, like:
        [
        ("a", "ey1"),
        ("b", "b_iy1"),
        ("c", "s_iy1"),
        ]
    :return: a string to be writen to the output file, like:
        abc ey1 b_iy1 s_iy1
    """
    word_list = []
    phones = []
    for step_tuple in path_list:
        word_list.append(step_tuple[0])
        phones.append(step_tuple[1])
        pass
    result = "".join(word_list) + "\t" + " ".join(phones) + "\n"
    return result

class Aligner:

    def __init__(self, training_file_name, test_data_file_name):
        self.training_data_file_name = training_file_name
        self.test_data_file_name = test_data_file_name
        self.transcription_list = list()
        self.grapheme_dict = dict()
        self.phoneme_dict = dict()
        self.prob_matrix = np.zeros(shape=(1, 1))
        pass

    def init_prob_matrix(self):
        """

        :return: matrix containing probabilities of a grapheme match a phoneme, initialized with 0 value.
        """
        g_count = len(self.grapheme_dict)
        p_count = len(self.phoneme_dict)
        self.prob_matrix = np.zeros(shape=(g_count, p_count), dtype=np.float32)
        logging.debug("prob_matrix:")
        logging.debug(self.prob_matrix)
        return self.prob_matrix

    def reset_prob_matrix(self, align_paths):
        """

        :param align_paths: a list of step lists, like:
            [
                [
                    ("a", "ey1"),
                    ("b", "b_iy1"),
                    ...,
                    ("c", "s_iy1"),
                ],
                [
                    ("a", "ey1"),
                    ("b", "b_iy1"),
                    ...,
                    ("c", "s_iy1"),
                ],
                ...,
                [
                    ("a", "ey1"),
                    ("b", "b_iy1"),
                    ...,
                    ("c", "s_iy1"),
                ],
            ]
        :return:
        """
        # if not self.prob_matrix:
        #     self.init_prob_matrix()
        logging.debug("before reset prob matrix:")
        logging.debug(self.prob_matrix)
        for align_path in align_paths:
            for step in align_path:
                g_id = self.get_grapheme_id(step[0])
                p_id = self.get_phoneme_id(step[1])
                self.prob_matrix[g_id][p_id] += 1
                pass
            pass
        self.normalize_prob_matrix()
        logging.debug("after reset prob matrix:")
        logging.debug(self.prob_matrix)
        return self.prob_matrix

    def normalize_prob_matrix(self):
        """

        probability matrix is a matrix with shape: (grapheme_count, phoneme_count)
        normalization is to keep sum of each row in the matrix to 1
        :return: a normalized probability matrix.
        """
        shape = self.prob_matrix.shape
        sum_array = np.sum(self.prob_matrix, axis=1)
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.prob_matrix[i][j] /= sum_array[i]
                pass
            pass
        logging.debug("prob_matrix:")
        logging.debug(self.prob_matrix)
        return self.prob_matrix

    def get_grapheme_id(self, grapheme):
        g_id = self.grapheme_dict[grapheme]
        return g_id

    def get_phoneme_id(self, phoneme):
        p_id = self.phoneme_dict[phoneme]
        return p_id

    def distance(self, grapheme, phoneme):
        """

        :param grapheme: a string like: a
        :param phoneme: a string like: ey1
        :return: probability of grapheme match phoneme
        """
        g_id = self.get_grapheme_id(grapheme)
        p_id = self.get_phoneme_id(phoneme)
        distance = self.prob_matrix[g_id][p_id]
        return distance

    def init_prob_of_grapheme_match_phoneme(self):
        """

        initialize prob_matrix: the probability of G matching P
        count with DTW all possible G/P association for all possible epsilon positions in the phonetic
        :return: prob_matrix
        """
        self.transcription_list = load_transcription(training_data_file_name)
        self.grapheme_dict = load_grapheme_dict(self.transcription_list)
        self.phoneme_dict = load_phoneme_dict(self.transcription_list)
        self.init_prob_matrix()
        align_paths = []
        for (word, phones) in self.transcription_list:
            pair_list = introduce_epsilon_phone_seq(word, phones)  # Introduce epsilon into phone list
            for (w, p) in pair_list:
                # align_path, _ = self.dynamic_time_wrapping(w, p)
                align_path = []
                for i in range(len(w)):
                    align_path.append((w[i], p[i]))
                align_paths.append(align_path)
            pass
        self.reset_prob_matrix(align_paths)
        return self.prob_matrix

    def dynamic_time_wrapping(self, word, phones):
        """

        :param word: a string represent a word
        :param phones: a list of string represent some phones
        :return: a list of tuple represent the best path, like:
            [
            ("a", "ey1"),
            ("b", "b_iy1"),
            ...,
            ("c", "s_iy1"),
            ]
        """
        g_count = len(word)
        p_count = len(phones)
        frame_dist_matrix = np.zeros(shape=(g_count, p_count), dtype=np.float32)  # Frame distance matrix.
        for i in range(g_count):
            for j in range(p_count):
                frame_dist_matrix[i][j] = self.distance(word[i], phones[j])
                pass
            pass
        acc_dist_matrix = np.zeros(shape=(g_count, p_count), dtype=np.float32)  # Accumulated distance matrix.
        acc_dist_matrix[0][0] = frame_dist_matrix[0][0]
        """Dynamic programming to compute the accumulated probability."""
        for i in range(1, g_count):
            for j in range(p_count):
                d1 = acc_dist_matrix[i-1][j]
                if j > 0:
                    d2 = acc_dist_matrix[i-1][j-1]
                else:
                    d2 = 0
                acc_dist_matrix[i][j] = frame_dist_matrix[i][j] + max([d1, d2])
                pass
            pass
        prob_value = acc_dist_matrix[g_count-1][p_count-1]
        """Trace back to find the best path with the max accumulated probability."""
        align_path = []
        #############################
        i, j = g_count-1, p_count-1
        while 1:
            align_path.append((word[i], phones[j]))
            if i == 0 & j == 0:
                break
            if i > 0:
                d1 = acc_dist_matrix[i - 1][j]
                if j > 0:
                    d2 = acc_dist_matrix[i - 1][j - 1]
                else:
                    d2 = 0
            else:
                d1 = 0
                d2 = 0
            # if j > 0:
            #     d3 = acc_dist_matrix[i][j - 1]
            # else:
            #     d3 = 0
            candidate_steps = [(i-1, j), (i-1, j-1)]
            candidate_prob = [d1, d2]
            i, j = candidate_steps[candidate_prob.index(max(candidate_prob))]
            pass
        align_path.reverse()
        ##########################
        # g_array, p_array = self._traceback(acc_dist_matrix)
        # for i in range(len(g_array)):
        #     if i > 0 & p_array[i] == p_array[i-1]:
        #         align_path.append((word[g_array[i]], "*"))
        #     else:
        #         align_path.append((word[g_array[i]], phones[p_array[i]]))
        #     pass
        return align_path, prob_value

    def _traceback(self, acc_dist_matrix):
        i, j = np.array(acc_dist_matrix.shape) - 2
        g, p = [i], [j]
        while i > 0 or j > 0:
            tb = np.argmax((acc_dist_matrix[i, j], acc_dist_matrix[i, j + 1], acc_dist_matrix[i + 1, j]))
            if tb == 0:
                i -= 1
                j -= 1
            elif tb == 1:
                i -= 1
            else:  # (tb == 2):
                j -= 1
            g.insert(0, i)
            p.insert(0, j)
        return np.array(g), np.array(p)

    def e_step(self):
        """

        :return:
        """
        align_paths = []
        for (word, phones) in self.transcription_list:
            pair_list = introduce_epsilon_phone_seq(word, phones)
            logging.debug("pair list:")
            logging.debug(pair_list)
            candidate_path_list = []  # Construct a candidate path list for all word-phones
            for (w, p) in pair_list:
                align_path, prob_value = self.dynamic_time_wrapping(w, p)
                candidate_path_list.append((align_path, prob_value))
            candidate_path_list.sort(key=lambda x: x[1], reverse=True)  # Sort by probability
            align_paths.append(candidate_path_list[0][0])  # Pick up the promising path with the biggest probability.
            pass
        return align_paths

    def m_step(self, align_paths):
        self.reset_prob_matrix(align_paths)
        pass

    def train(self, iter_num, epsilon):
        """

        train prop matrix until iter_num or the difference of adjacent iteration results is no more than epsilon
        :param iter_num:
        :param epsilon:
        """
        self.init_prob_of_grapheme_match_phoneme()
        for i in range(iter_num):
            logging.info("Training epoch:" + str(i))
            last_prob_matrix = self.prob_matrix.copy()
            align_paths = self.e_step()  # Expectation step
            self.m_step(align_paths)  # Maximum step
            # if self.is_prob_matrix_equal(last_prob_matrix, self.prob_matrix, epsilon):
            #     break
            pass
        pass

    def align(self):
        transcription_list = load_transcription(self.test_data_file_name)
        result_list = []
        for (word, phones) in transcription_list:
            pair_list = introduce_epsilon_phone_seq(word, phones)
            candidate_path_list = []  # Construct a candidate path list for all possible word-phones pairs
            for (w, p) in pair_list:
                align_path, prob_value = self.dynamic_time_wrapping(w, p)
                candidate_path_list.append((align_path, prob_value))
            candidate_path_list.sort(key=lambda x: x[1], reverse=True)  # Sort by probability
            result_string = path_to_string(candidate_path_list[0][0])
            result_list.append(result_string)  # Pick up the promising path with the biggest probability.
        with open(output_file_name, "w") as output_file:
            output_file.writelines(result_list)
            pass
        pass
    pass


if __name__ == '__main__':
    iter_num = 5
    epsilon = 0
    aligner = Aligner(training_data_file_name, test_data_file_name)
    aligner.train(iter_num, epsilon)
    aligner.align()
