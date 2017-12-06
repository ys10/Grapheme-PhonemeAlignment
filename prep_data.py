# -*- coding: utf-8 -*-

from cmudict import CMUDict


if __name__ == '__main__':
    dct = CMUDict('data/cmudict-0.7b')
    for word in dct.words():
        print(word)
        print(dct[word])


