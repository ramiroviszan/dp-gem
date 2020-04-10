from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Flatten, Lambda


def create_model(key, params):
    return models[key](*params) 

def create_control_model(vocab_size):
    #The LSTM input layer must be 3D.
    #The meaning of the 3 input dimensions are: samples, time steps, and features.
    #The LSTM input layer is defined by the input_shape argument on the first hidden layer.
    #The input_shape argument takes a tuple of two values that define the number of time steps and features.
    #The number of samples is assumed to be 1 or more.
    #Either input_shape(length=None or window_size for fixed length, input_dim=vocab_size,)...
    ####...if symbols in each x of X is a onehot vector [[0 0 0 1] [1 0 0 0] [0 0 0 1]]
    ####...In this case X for train must be (batch, window_size, vocab_size)
    #OR input_shape(length=None or window_size, input_dim=1,)...
    ###...if X is a vector of integer symbols [4 1 4].
    ###...In this chase X for train must be (batch, window_size, 1) and np.expand_dims(train_x, axis=3) is needed
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(None, 1,)))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    return model

def create_utility_model(vocab_size):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(None, 1,)))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    return model

def create_dp_gen_model(vocab_size, emb_size, context_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=emb_size, input_length=context_size, mask_zero=False))
    model.add(Flatten())
    model.add(Dense(vocab_size, activation='softmax', kernel_regularizer='l2'))

    return model


models = {
    'control': create_control_model,
    'utility': create_utility_model,
    'gen': create_dp_gen_model
}