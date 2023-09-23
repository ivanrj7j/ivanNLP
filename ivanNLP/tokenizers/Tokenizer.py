from typing import Union
from ivanNLP.tokens import Token

class Tokenizer:
    """
    Base class for all the tokenizer classes
    """
    def __init__(self) -> None:
        pass

    def fit(self, corpus:Union[str, list[str]]):
        """
        Extracts all the tokens from the corpus
        """

        raise NotImplementedError("This method should be implemented by the child")
    
    def tokenize(self, token:str) -> Token:
        """
        Gives the token object for the given token
        """
        
        raise NotImplementedError("This method should be implemented by the child")