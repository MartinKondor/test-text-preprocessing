import re
from typing import List, Union

import numpy as np
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from src import data

from src.data import load_data


stopwords_set = list(set(stopwords.words('english')))


def tokenization(sentences: np.ndarray) -> np.ndarray:
    tokenizer = Tokenizer(lower=False, oov_token="<NULL>")
    tokenizer.fit_on_texts(sentences)
    return tokenizer.texts_to_sequences(sentences)


def tokenization_with_prep(sentences: np.ndarray) -> np.ndarray:
    
    def prep(sentence: str) -> str:
        return sentence.strip().lower()

    tokenizer = Tokenizer(lower=True, oov_token="<NULL>")
    tokenizer.fit_on_texts(map(prep, sentences))
    return tokenizer.texts_to_sequences(sentences)


def tokenization_without_stopwords(sentences: np.ndarray) -> np.ndarray:
    
    def prep(sentence: str) -> str:
        sentence = sentence.strip().lower()

        # Remove multiple spaces
        sentence = " ".join([l for l in sentence.split(" ") if l])
        return " ".join([w for w in sentence.split(" ") if len(w) > 1 and (w not in stopwords_set)])

    tokenizer = Tokenizer(lower=True, oov_token="<NULL>")
    tokenizer.fit_on_texts(map(prep, sentences))
    return tokenizer.texts_to_sequences(sentences)


def prep_text_with_regex(sentences: np.ndarray) -> np.ndarray:
    
    def prep(_sentence: str) -> str:
        sentence = _sentence.strip().lower()
        sentence = sentence.replace(".", "")
        sentence = sentence.replace("?", "")
        sentence = sentence.replace("!", "")
        sentence = sentence.replace(",", "")

        # Replace numbers and special characters
        sentence = re.sub("[0-9]+", "x", sentence)
        sentence = re.sub("[^a-zA-Z0-9öüóőúéáűíÖÜÓŐÚÉÁŰÍ\s]", "", sentence)

        # Remove multiple spaces
        sentence = " ".join([l for l in sentence.split(" ") if l])
        return " ".join([w for w in sentence.split(" ") if len(w) > 1 and (w not in stopwords_set)])


    tokenizer = Tokenizer(lower=True, oov_token="<NULL>")
    tokenizer.fit_on_texts(map(prep, sentences))
    return tokenizer.texts_to_sequences(sentences)


if __name__ == "__main__":
    np.seterr(all="ignore")
    np.random.seed(0)
    dataset = load_data()["title"].sample(frac=1).values
    test_sentences = dataset[:5]

    print("[test_sentences]")
    for i, test_sentence in enumerate(test_sentences):
        print(f"{i}\t--->", test_sentence)
    print()

    results = [
        ["tokenization", tokenization(test_sentences)],
        ["tokenization_with_prep", tokenization_with_prep(test_sentences)],
        ["tokenization_without_stopwords", tokenization_without_stopwords(test_sentences)],
        ["prep_text_with_regex", prep_text_with_regex(test_sentences)],
    ]


    def count_params(sentences: np.ndarray) -> int:
        elements = []
        for sentence in sentences:
            elements = np.unique([*elements, *np.unique(sentence)])
        return len(elements)


    for name, result in results:
        print(f"[{name}]")

        for i, r in enumerate(result):
            print(f"{i}\t--->", r)

        print(f"[{name}] number of unique parameters:", count_params(result))
        print()
