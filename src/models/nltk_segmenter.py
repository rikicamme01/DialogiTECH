import nltk
nltk.download('punkt')
from nltk import sent_tokenize

# wrapper sentence_tokenizer nltk
class NLTKSegmenter():
    
    def predict(self, text: str) -> list[str]:
        return sent_tokenize(text, language='italian')

