from typing import List
from experiment.eval.bleu import Bleu

def eval_scores(
    hypothesis: List[str], 
    references: List[List[str]],
    metric
    ):
    if metric == 'bleu':
        scorer = Bleu(n_gram=1)
    else:
        raise ValueError

    normal_score, average_score = scorer.compute_score(hypothesis, references)
    return normal_score, average_score

if __name__ == '__main__':
    ALL_METRICS = ['bleu']

    hypothesis_corpus = 'I think I can do it'.split()
    references_corpus = [
        'I think I can do it'.split(), 
        'You are idiot'.split()
    ]

    print(hypothesis_corpus)

    for metric in ALL_METRICS:
        normal_score, average_score = eval_scores(hypothesis_corpus, references_corpus, metric)
        print(normal_score)
        print(average_score)