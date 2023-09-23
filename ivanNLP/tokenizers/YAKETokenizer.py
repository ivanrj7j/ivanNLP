from typing import Union
from ivanNLP.tokenizers import Tokenizer
from ivanNLP.tokens import Token
from yake import KeywordExtractor
import yake
from numpy import array, float32


english_stopwords = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "via", "also", 
    "around"
}


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