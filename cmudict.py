# -*- coding: utf-8 -*-

import codecs
from collections import defaultdict
from collections import Counter


class CMUDict(object):
    def __init__(self, filename):
        # load CMU dict
        with codecs.open(filename, encoding='utf-8') as fin:
            lines = (line for line in fin if line.strip() and not line.startswith(';;;'))
            lines = (line.strip().lower().split(maxsplit=1) for line in lines)

            # merge phonemes for the same word
            self.data = defaultdict(list)
            for word, phone in lines:
                if word.endswith(('(1)', '(2)', '(3)')):
                    original_word = word[:-3]
                    self.data[original_word].append(phone)
                else:
                    self.data[word].append(phone)

    def words(self):
        return self.data.keys()

    def symbols(self):
        freqs = Counter()
        for _, phones_list in self.data.items():
            for phones in phones_list:
                freqs.update(phones.split())

        return freqs

    def alignment(self):
        pass

    def __getitem__(self, item):
        return self.data[item]



