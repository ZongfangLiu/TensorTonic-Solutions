import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model):
            if i % 2 == 0:
                pe[pos, i] = np.sin(pos / (base ** (i / d_model)))
            else:
                pe[pos, i] = np.cos(pos / (base ** ((i - 1) / d_model)))
    return pe