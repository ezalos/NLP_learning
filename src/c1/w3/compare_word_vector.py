import numpy as np
from .cosine_similarity import cosine_similarity

def get_B2_from_B1_like_A1_to_A2(A1, B1, A2, embeddings, cosine_similarity=cosine_similarity):
    """
    Input:
        A1: a string (the capital city of country1)
        country1: a string (the country of capital1)
        B1: a string (the capital city of country2)
        embeddings: a dictionary where the keys are words and values are embeddings
    Output:
        countries: a dictionary with the most likely country and its similarity score
    """
    group = set([A1, B1, A2])

    A1_emb = embeddings[A1]
    B1_emb = embeddings[B1]
    A2_emb = embeddings[A2]

    vec = (B1_emb - A1_emb) + A2_emb

    similarity = -1
    B2 = ''

    for word in embeddings.keys():
        if word not in group:
            word_emb = embeddings[word]
            cur_similarity = cosine_similarity(vec, word_emb)
            if cur_similarity > similarity:
                similarity = cur_similarity
                B2 = (word, similarity)
    return B2
