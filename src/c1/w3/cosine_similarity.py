import numpy as np

def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    dot = np.dot(A, B)
    norma = np.sqrt(np.dot(A, A))
    normb = np.sqrt(np.dot(B, B))
    cos = dot / (norma * normb)
    return cos

