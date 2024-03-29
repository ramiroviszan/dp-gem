import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Flatten, Lambda, RepeatVector, Input, Add, Dropout
from tensorflow.keras.utils import CustomObjectScope

from common.tensorflow.norm_clipping import Norm1Clipping, Norm2Clipping

def create_model(key, params):
    return models[key](*params) 

def create_control_model(vocab_size, window_size, emb_size, drop_out, hidden_layers=[256]):
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
    ###...In this chase X for train must be (batch, window_size, 1) and np.expand_dims(train_x, axis=2) is needed
    model = Sequential()
    
    model.add(Embedding(input_dim=vocab_size, output_dim=emb_size, input_length=window_size, mask_zero=True))
    for size in hidden_layers:
            model.add(LSTM(size, return_sequences = True))
    
    model.add(Dropout(drop_out))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))

    return model

#gen autoencoder lap
def create_dp_gen_lap_autoencoder_model(vocab_size, max_length, emb_size, hidden_state_size):
    inputSeq = Input(shape=(max_length,))
    inputNoise = Input(shape=(hidden_state_size,))
    x = Embedding(vocab_size, emb_size, input_length=max_length, mask_zero=True)(inputSeq)
    x = LSTM(hidden_state_size, return_state=False, return_sequences=False)(x)
    x = Norm1Clipping()(x)
    x = Add()([x, inputNoise])
    x = RepeatVector(max_length)(x)
    x = LSTM(hidden_state_size, return_sequences=True)(x)
    x = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)

    model = Model(inputs=[inputSeq, inputNoise], outputs=x)
    return model


def load_model_adapter(path):
    with CustomObjectScope({'Norm1Clipping': Norm1Clipping, 'Norm2Clipping': Norm2Clipping}):
        return load_model(path)

models = {
    'control': create_control_model,
    'gen': create_dp_gen_lap_autoencoder_model
}