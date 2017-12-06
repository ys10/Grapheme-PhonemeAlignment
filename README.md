# English LTS

English letter to sound tool

## cmudict

- http://www.speech.cs.cmu.edu/cgi-bin/cmudict

## cmudict with syllables

- http://webdocs.cs.ualberta.ca/~kondrak/cmudict.html
- https://en.wikipedia.org/wiki/Syllabification

## wagon CART tree

- http://www.cstr.ed.ac.uk/projects/speech_tools/manual-1.2.0/x3475.htm

## Grapheme-Phoneme Alignment
This project aims to implement a algorithm proposed in paper as follows:

[Pagel V, Lenzo K, Black A. Letter to Sound Rules for Accented Lexicon Compression[J]. 1998, cmp-lg/9808010.](https://arxiv.org/abs/cmp-lg/9808010)

### Project Structure
* **g2p_data_pre_process.py** - a python script to pre-process g2p data by rules.
* **g2p_aligner.py** - a python script to train model by EM & DTW algorithm.
* **assets/** - store the training data sets.
* **log/** - settle the log files.

### G2P Data Preprocess
Try to match every word-phones pair to every rule based on the locations of grapheme & phoneme.
#### how to match word-phones pair to a rule.
1. Match a word-phones pair to a rule based on the locations of grapheme & phoneme.
2. If a rule got matched, modify the phones and rematch the word-phones pair to every rule, until no rule would match.
3. Set a admissible error to admit some error caused by previous silent graphemes(letters).
