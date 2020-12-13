import os
import sys
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras
from tensorflow.keras.models import load_model
from datagen import load_data
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input folder, consisting training and validation data" , default=".")
    parser.add_argument("-l", "--logdir", type=str, help="TensorBoard logging directory", default="logs/scalars")
    parser.add_argument("-e", "--earlystopping", type=int, help="Early stopping patience", default=10)
    parser.add_argument("-m", "--model", type=str, help="Model path", default='models/')
    parser.add_argument("-c", "--checkpoints", type=bool, help="Use checkpoints (save best model during training)", default=False)
    parser.add_argument("-o", "--output", type=str, help="Trained model is saved to here", default="trained")

    args = parser.parse_args()
    for arg in vars(args):
        if getattr(args, arg) is None:
            parser.print_help()
            sys.exit(1)

    x_train, y_train, x_valid, y_valid, _, _ = load_data(args.input)
    model = train_model(x_train, y_train, x_valid, y_valid, args.model, args.checkpoints, args.logdir)
    model.save(args.output)

def train_model(x_train, y_train, x_valid, y_valid, model_path, use_checkpoints, logdir):
    # Define callbacks
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(logdir, now)
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 save_best_only=True, verbose=1)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75,
                                  patience=10, min_lr=0.0001, verbose=1)
    early_stopping = EarlyStopping(patience=20, verbose=1)
    model = load_model(model_path)
    model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              batch_size=512, epochs=3000,
              callbacks=list(filter(None, [tensorboard_callback, early_stopping, checkpoint if use_checkpoints else None, reduce_lr])), verbose=2)
    return model


if __name__ == '__main__':
    main()