from numpy import ndarray

class Token:
    def __init__(self, token:str, vector:ndarray) -> None:
        self.token = token
        self.vector = vector

    def __str__(self) -> str:
        return self.token
    
    def __repr__(self) -> str:
        return self.token