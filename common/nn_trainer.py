from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from wandb.keras import WandbCallback

import common.plot_utils as plot_utils


class NNTrainer:

    def train(self, model, save_path, train_x, train_y, train_sessions, use_wandb=False):

        for key in train_sessions.keys():

            print("\nTrain Session:", key)
            session_info = train_sessions[key]
            epochs, batch_size, lr, loss, val_split, patience, save_model = session_info.values()
            model.compile(loss=loss, optimizer=Adam(lr=lr))

            earlystopper = EarlyStopping(
                monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
            
            if use_wandb:
                logger = WandbCallback()
                callbacks = [earlystopper, logger]
            else:
                callbacks = [earlystopper]

            history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                                validation_split=val_split, callbacks=callbacks, verbose=1)

            plot_utils.plot(history, None, save_path + "_session_" + key)
            if save_model:
                model.save(save_path)

        return model
