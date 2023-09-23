from typing import Union
from ivanNLP.tokenizers import Tokenizer
from ivanNLP.tokens import Token
from numpy import array

class WordTokenizer(Tokenizer):
    """
    Converts every word into a token, this uses .split() method to tokenize words
    """
    def __init__(self) -> None:
        super().__init__()

    def fit(self, corpus: str | list[str]):
        def tokenize(c:str):
            words = set(corpus.lower().split())

            tokens = [Token(x, array([])) for x in words]

            self.vocabulary = tokens


        if isinstance(corpus, str):
            tokenize(corpus)
        elif isinstance(corpus, tuple) or isinstance(corpus, list):
            c = " ".join(corpus)
            tokenize(c)
        else:
            raise TypeError("The given corpus should be a string or list of string or tuple of string")