from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def get_tokens(questions, answers):
    """
    Computes the tokens of a lists of questions and answers
    
    Returns:
    -- token_ques_input: tokens of questions to input to encoder
    -- token_ans_input: tokens of answers to input to decoder
    -- token_ans_target: tokens of answers as target to decoder
    -- vocab_size: size of the vocabulary (number of words)
    -- t: Tokenizer() object instance

    parameters:
    -- questions: list of strings (questions)
    -- answers: list of strings (answers)
    """

    t = Tokenizer()

    t.fit_on_texts(answers + questions)

    t.texts_to_matrix(answers, mode='count').shape

    vocab_size = len(t.word_index) + 1

    token_ans_target = [one_hot(sentence, n=len(t.word_index)) for sentence in answers]
    token_ans_input = [[vocab_size] + one_hot(sentence, n=len(t.word_index)) for sentence in answers]
    token_ques_input = [one_hot(sentence, n=len(t.word_index)) for sentence in questions]

    return token_ques_input, token_ans_input, token_ans_target, vocab_size, t


def padding(token_ques_input, token_ans_input, token_ans_target, max_len):
    """
    Pads the input sequences to the max_len
    
    Returns:
    -- pad_ques_input: padded tokens of questions to input to encoder
    -- pad_ans_input: padded tokens of answers to input to decoder
    -- pad_ans_target: padded tokens of answers as target to decoder

    parameters:
    -- token_ques_input: tokens of questions to input to encoder
    -- token_ans_input: tokens of answers to input to decoder
    -- token_ans_target: tokens of answers as target to decoder
    -- max_len: maximum length of the sequences
    """

    pad_ans_target = pad_sequences(token_ans_target, maxlen=max_len, dtype='int32', padding='post', truncating='post', value=0)
    pad_ans_input = pad_sequences(token_ans_input, maxlen=max_len, dtype='int32', padding='post', truncating='post', value=0)
    pad_ques_input = pad_sequences(token_ques_input, maxlen=max_len, dtype='int32', padding='post', truncating='post', value=0)

    return pad_ques_input, pad_ans_input, pad_ans_target


def one_hot_encode(pad_ques_input, pad_ans_input, pad_ans_target, vocab_size):
    """
    Pads the input sequences to the max_len
    
    Returns:
    -- pad_ques_input: one hot encoded padded tokens of questions to input to encoder
    -- pad_ans_input: one hot encoded padded tokens of answers to input to decoder
    -- pad_ans_target: one hot encoded padded tokens of answers as target to decoder

    parameters:
    -- pad_ques_input: padded tokens of questions to input to encoder
    -- pad_ans_input: padded tokens of answers to input to decoder
    -- pad_ans_target: padded tokens of answers as target to decoder

    """

    pad_ans_target = to_categorical(pad_ans_target, num_classes=vocab_size+1)
    pad_ans_input = to_categorical(pad_ans_input, num_classes=vocab_size+1)
    pad_ques_input = to_categorical(pad_ques_input, num_classes=vocab_size+1)

    return pad_ques_input, pad_ans_input, pad_ans_target
