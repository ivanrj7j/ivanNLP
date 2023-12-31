from typing import Union
from ivanNLP.tokens import Token
from numpy import ndarray, stack, array_equal
from string import punctuation

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
    
    @property
    def tokenMatrix(self) -> ndarray:
        vectors = [x.vector for x in self.vocabulary]

        return stack(vectors)
    
    @property
    def keywords(self) -> list[str]:
        return [x.token for x in self.vocabulary]
    
    @staticmethod
    def removePunctuations(doc:str):
        for char in punctuation:
            doc = doc.replace(char, '')

        return doc
    
    def getToken(self, token:str) -> Token:
        """
        Gives the token object for the given token
        """
        results = tuple(filter(lambda x: x.token == token, self.vocabulary))

        if len(results) == 0:
            raise ModuleNotFoundError("There are no token object for given token")
        
        return results[0]
    
    def getTokenFromVector(self, vector:ndarray):
        results = tuple(filter(lambda x: array_equal(x.vector, vector), self.vocabulary))

        if len(results) == 0:
            raise ModuleNotFoundError("There are no token with the given vector")
        
        return results[0]