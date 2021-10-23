import os
import pickle


class LanguageModel:
    """Simple character-level language model"""

    def __init__(self) -> None:
        self._unigram = {}
        self._bigram = {}

    def get_char_unigram(self, c: str) -> float:
        """Probability of character c."""
        return self._unigram[c]

    def get_char_bigram(self, c: str, d: str) -> float:
        """Probability that character c is followed by character d."""
        return self._bigram[c][d]

    def train(self, txt: str, chars: str):
        """Create language model from text corpus."""
        # compute unigrams
        self._unigram = {c: 0 for c in chars}
        for c in txt:
            # ignore unknown chars
            if c not in self._unigram:
                continue
            self._unigram[c] += 1

        # compute bigrams
        self._bigram = {c: {d: 0 for d in chars} for c in chars}
        for i in range(len(txt) - 1):
            c = txt[i]
            d = txt[i + 1]

            # ignore unknown chars
            if c not in self._bigram or d not in self._bigram[c]:
                continue

            self._bigram[c][d] += 1

        # normalize
        sum_unigram = sum(self._unigram.values())
        for c in chars:
            self._unigram[c] /= sum_unigram

        for c in chars:
            sum_bigram = sum(self._bigram[c].values())
            if sum_bigram == 0:
                continue
            for d in chars:
                self._bigram[c][d] /= sum_bigram

    def save(self, path):
        with open(os.path.join(path, "unigram.pkl"), 'wb') as pkl:
            pickle.dump(self._unigram, pkl)
        with open(os.path.join(path, "bigram.pkl"), 'wb') as pkl:
            pickle.dump(self._bigram, pkl)

    def load(self, path):
        with open(os.path.join(path, "unigram.pkl"), 'rb') as pkl:
            self._unigram = pickle.load(pkl)
        with open(os.path.join(path, "bigram.pkl"), 'rb') as pkl:
            self._bigram = pickle.load(pkl)
