from typing import Union
from ivanNLP.tokenizers import Tokenizer
from ivanNLP.tokens import Token
from yake import KeywordExtractor
import yake
from numpy import array, float32
from ivanNLP.stopwords import english_stopwords


class YAKETokenizer(Tokenizer):
    """
    Uses YAKE(Yet another keyword extractor) to extract tokens
    """
    def __init__(self) -> None:
        super().__init__()

    def fit(self, corpus: str | list[str]):
        def tokenize(c:str):
            extractor = KeywordExtractor(top=float('inf'), stopwords=english_stopwords)

            tokens = []

            for keyword, score in extractor.extract_keywords(c):
                tokens.append(Token(keyword, array([score], dtype=float32)))

            self.vocabulary = tokens


        if isinstance(corpus, str):
            tokenize(corpus.lower())
        elif isinstance(corpus, tuple) or isinstance(corpus, list):
            c = " ".join(corpus)
            tokenize(c.lower())
        else:
            raise TypeError("The given corpus should be a string or list of string or tuple of string")