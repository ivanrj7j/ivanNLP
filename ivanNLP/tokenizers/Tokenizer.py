from typing import Union
from ivanNLP.tokens import Token
from numpy import ndarray

class Tokenizer:
    """
    Base class for all the tokenizer classes
    """
    def __init__(self) -> None:
        self.vocabulary:list[Token] = []

    def fit(self, corpus:Union[str, list[str]]):
        """
        Extracts all the tokens from the corpus
        """

        raise NotImplementedError("This method should be implemented by the child")
    
    def getToken(self, token:str) -> Token:
        """
        Gives the token object for the given token
        """
        
        raise NotImplementedError("This method should be implemented by the child")
    
    def getTokenFromVector(self, vector:ndarray):
        results = tuple(filter(lambda x: x.vector == vector, self.vocabulary))

        if len(results) == 0:
            raise ModuleNotFoundError("There are no token with the given vector")
        
        return results[0]