from typing import Union
from ivanNLP.tokenizers import Tokenizer
from ivanNLP.tokens import Token
from yake import KeywordExtractor
from numpy import array, float32
from ivanNLP.stopwords import english_stopwords

class WordYakeHybrid(Tokenizer):
    def __init__(self, multiWordPercentage:float=0.2) -> None:
        """
        Initializes the object
        
        Keyword arguments:
        multiWordPercentage -- The fraction of the total tokens to be multi word
        Return: None
        """
        
        super().__init__()
        self.multiWordPercentage = multiWordPercentage

    def fit(self, corpus: str | list[str]):
        def tokenize(c:str):
            c = Tokenizer.removePunctuations(c)
            
            words = set(c.lower().split())

            tokens = {Token(word, array([0])) for word in words}

            extractor = KeywordExtractor(top=round(len(words)*self.multiWordPercentage), stopwords=english_stopwords)

            for keyword, score in extractor.extract_keywords(c):
                tokens.add(Token(keyword, array([score], dtype=float32)))

            self.vocabulary = list(tokens)


        if isinstance(corpus, str):
            tokenize(corpus.lower())
        elif isinstance(corpus, tuple) or isinstance(corpus, list):
            c = " ".join(corpus)
            tokenize(c.lower())
        else:
            raise TypeError("The given corpus should be a string or list of string or tuple of string")