import numpy as np

def generate_n_names(n, max_len, char_to_idx, model):
    """
    Generate n names automatically
    
    Returns:

    parameters:
    -- n: the number of names to generate (int)
    -- max_len: the length of the sequence
    -- char_to_idx: the dict giving the char corresponding to each idx
    -- model: the trained model that will be used to generate names
    """
    for _ in range(n):
        stop=False
        ch='\t'
        counter=1
        target_seq = np.zeros((1, max_len, len(char_to_idx)))
        target_seq[0, 0, char_to_idx[ch]] = 1.
        while stop == False and counter < 10:
            #sample the data
            probs = model.predict_proba(target_seq, verbose=0)[:,counter-1,:]
            c = np.random.choice(list(char_to_idx.keys()), replace =False, p=probs.reshape(len(char_to_idx)))
            if c=='\n':
                stop=True
            else:
                ch=ch+c
                target_seq[0, counter ,char_to_idx[c]] = 1.
                counter=counter+1
        print(ch)
