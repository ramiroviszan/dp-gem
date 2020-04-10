from keras.optimizers import Adam
import plot_utils

class NNTrainer:

    def train(self, model, save_path, train_x, train_y, train_sessions):

        for key in train_sessions.keys():
            print("\nTrain Session:", key)
            session_info = train_sessions[key]
            epochs, batch_size, lr, loss, val_split = session_info.values()
            model.compile(loss=loss, optimizer= Adam(lr=lr))
            history = model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, validation_split=val_split, verbose=1)
            plot_utils.plot(history, None, save_path)

        model.save(save_path)
        return model