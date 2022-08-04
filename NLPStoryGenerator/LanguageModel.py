from collections import defaultdict
from collections import Counter
import numpy as np

def unzip(pairs):
    return tuple(zip(*pairs))

def normalize(counter):
        total = sum(counter.values())
        return [(char, cnt/total) for char, cnt in counter.most_common()]

class LanguageModel:
    def __init__(self, text, n):
        self.n = n
        raw_lm = defaultdict(Counter) # <COGSTUB> history -> {char -> count}
        history = "~" * (n - 1)  # <COGSTUB> length n - 1 history
        
        # count number of times characters appear following different histories
        #
        # for char in text ...
        #    1. Increment language model's count, given current history and character
        #    2. Update history

        # <COGINST>
        
        for word in text:
            raw_lm[history][word] += 1
            
            # slide history window to the right by one character
            history = history[1:] + word
        # </COGINST>
        
        # create the finalized language model – a dictionary with: history -> [(char, freq), ...]
        self.lm = {history : normalize(counter) for history, counter in raw_lm.items()}  # <COGSTUB>
    
    def generate_letter(self, history):
        # <COGINST>
        if not history in self.lm:
            return "~"
        letters, probs = unzip(self.lm[history])
        i = np.random.choice(letters, p=probs)
        return i
    
    def generate_text(self, history, nletters=100):
        history= history[-(self.n - 1):]
        og_history = history
        text = []
        for _ in range(nletters):
            c = self.generate_letter(history)
            text.append(c)
            history = history[1:] + c
        return og_history + "".join(text)  
        # </COGINST>
