from .classify_abstruct_or_concrete import classify

def get_sconj_like_range(doc):
    sconj_like_range = []
    is_started = False

    for token in doc:
        if is_started == False and str(token.text).lower() == "like" and token.pos_ == "SCONJ": 
            sconj_like_range.append(token.i)
            is_started = True
        elif is_started == True and token.pos_ == "PUNCT": 
            sconj_like_range.append(token.i)
            is_started = False

    if len(sconj_like_range) % 2 == 1: sconj_like_range.append(len(doc))

    return sconj_like_range

def is_remind(noun_chunk, token_dic):
    root_noun = token_dic[noun_chunk.root.i]
    dep_of_root_noun =  token_dic[root_noun[4]]

    if dep_of_root_noun[1] != "of":
        return False

    verb_of_root_noun = token_dic[dep_of_root_noun[4]]
    if verb_of_root_noun[1] in ['remind', 'reminds']:
        return True

    return False

def is_valid(noun_chunk, sconj_like_range, token_dic, classifier):
    PRP_LIST = ['i', 'my', 'me', 'mine', 'you', 'your', 'yours', 'she', 'her', 'hers', 'he', 'his', 'him', 'they', 'their', 'them', 'theirs', 'we', 'our', 'us', 'ours', 'this', 'that', 'it']
    QUESTION_LIST = ['which', 'what', 'why', 'how', 'where', 'when', 'who', 'whatever', 'whenever', 'wherever', 'whoever']
    OTHER_LIST = ['all', 'some', 'any', 'something', 'anything', 'everything']
    SPECIAL_LIST = ['evokes']
    invalid_list = PRP_LIST + QUESTION_LIST + OTHER_LIST + SPECIAL_LIST

    if str(noun_chunk).lower() in invalid_list:
        return False

    if str(noun_chunk.root.head.text).lower() == "like" and noun_chunk.root.head.pos_ == "ADP":
        return False

    if sconj_like_range != [] and sconj_like_range[0] < noun_chunk.root.i < sconj_like_range[1]:
        return False

    if is_remind(noun_chunk, token_dic): 
        return False

    if classify(classifier, noun_chunk.root) == "abstract":
        return False

    return True
