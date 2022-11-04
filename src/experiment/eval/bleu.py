from typing import List
from nltk import bleu_score

class Bleu():
    def __init__(self, n_gram: int):
        self.n_gram = n_gram

    def compute_score(
        self,
        hypothesis: List[str], 
        references: List[List[str]],
        ):
        fn = bleu_score.SmoothingFunction().method3
        normal_score = bleu_score.sentence_bleu(references, hypothesis, smoothing_function=fn)

        total_scores = 0
        for reference in references:
            total_scores += bleu_score.sentence_bleu(reference, hypothesis, smoothing_function=fn)
        
        return normal_score, total_scores/len(references)

