from experiment.eval.bleu import Bleu

def eval_scores(
    hypothesis, # List[str]
    references, # List[List[str]]
    metric
    ):
    if metric == 'bleu':
        scorer = Bleu(n_gram=1)
    else:
        raise ValueError

    normal_score, average_score = scorer.compute_score(hypothesis, references)
    return normal_score, average_score

def batch_eval_scores(
    batch_hypothesis, # List[List(str)]
    batch_references, # List[List[List(str)]]
    metric
    ):
    scorer = choose_metric(metric)

    total_normal_score = 0
    total_average_score = 0

    for i in range(len(batch_hypothesis)):
        hypothesis = batch_hypothesis[i]
        references = batch_references[i]
        normal_score, average_score = scorer.compute_score(hypothesis, references)
        total_normal_score += normal_score
        total_average_score += average_score

    return total_normal_score / len(batch_hypothesis), total_average_score / len(batch_hypothesis)

def choose_metric(metric):
    if metric == 'bleu':
        return Bleu(n_gram=1)
    else:
        raise ValueError

if __name__ == '__main__':
    ALL_METRICS = ['bleu']

    hypothesis1 = 'I think I can do it'.split()
    hypothesis2 = 'I hate you'.split()
    hypothesis3 = 'He run very fast'.split()

    references1 = [
        'I think I can do it'.split(),
        'You are idiot'.split(),
        'Watching TV is very fun'.split()
    ]

    references2 = [
        'I want to watch movie'.split(),
        'I have to win this game'.split(),
        'I hate you'.split(),
    ]

    references3 = [
        'I am going to go to school'.split(),
        'He run very fast'.split(),
        'Thank you very much'.split(),
    ]

    batch_hypothesis = [hypothesis1, hypothesis2, hypothesis3]
    batch_references = [references1, references2, references3]

    for metric in ALL_METRICS:
        normal_score, average_score = batch_eval_scores(batch_hypothesis, batch_references, metric)
        print(normal_score)
        print(average_score)