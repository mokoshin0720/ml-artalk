def dataframes_to_coco_eval_format(references, hypothesis):
    references = {i: [k for k in x] for i, x in enumerate(references)}
    hypothesis = {i: [x] for i, x in enumerate(hypothesis)}
    return references, hypothesis

if __name__ == '__main__':
    candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
    references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]