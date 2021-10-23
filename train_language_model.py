import os
import argparse
from asr.language_model import LanguageModel
from asr.wav2vec2.vocab import vocab_list
from tqdm import tqdm

corpus_path = os.path.join("data", "lm_training_corpus", "corpus.txt")
save_path = os.path.join("data", "models", "lm")
parser = argparse.ArgumentParser(description="Train character language model from text corpus")
parser.add_argument("--corpus", "-c", default=corpus_path, type=str, help="path to text corpus for training")
parser.add_argument("--save", "-s", default=save_path, type=str, help="path to save trained model")
args = parser.parse_args()


def change_digit_to_word(x):
    x = x.replace("0", "zero ")
    x = x.replace("1", "one ")
    x = x.replace("2", "two ")
    x = x.replace("3", "three ")
    x = x.replace("4", "four ")
    x = x.replace("5", "five ")
    x = x.replace("6", "six ")
    x = x.replace("7", "seven ")
    x = x.replace("8", "eight ")
    x = x.replace("9", "nine ")
    x = x.replace("  ", " ")
    x = x.strip()
    return x


# excluding pad token for language model
lm = LanguageModel(chars=vocab_list[1:])

lines = sum(1 for i in open(args.corpus, "r"))
with open(args.corpus, "r") as txt:
    for line in tqdm(txt, total=lines):
        line = change_digit_to_word(line)
        lm.train(line)
lm.normalize()
lm.save(args.save)