import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Input, Flatten,  Conv2D, MaxPooling2D
import warnings
import logging
warnings.filterwarnings('ignore')


DATA_INPUT_PATH = '/app/data/raw/Font Dataset Large'
MODEL_FILE_PATH = "/app/artifacts/final_model.h5"
IMG_SIZE = 224
NUM_COLOR_BITS = 3
BATCH_SIZE = 32
NUM_CLASSES = 48
TARGET_SIZE = (IMG_SIZE, IMG_SIZE)
MODEL_NAME = 'vgg16'
EPOCHS = 10

def get_model(model_input_shape, num_classes):
    base_model = VGG16
    base_model = base_model(include_top=False, weights="imagenet", input_shape = model_input_shape)
    base_model.trainable = False
    
    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation="softmax"))

    return model

class stopTrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > 0.95):
            print("\nReached 95% accuracy - stopping training")
            self.model.stop_training = True

def main():
    logger = logging.getLogger("Font type model trainer")
    logger.setLevel(logging.INFO)

    logger.info("Creating image data generator")
    datagen = ImageDataGenerator(
                                brightness_range=[0.2, 1.0],
                                # zoom_range = 0.2,
                                # shear_range = 0.15,
                                # horizontal_flip = True,
                                rescale = 1./255,
                                validation_split = 0.2)


    logger.info("Splitting data")
    train_generator = datagen.flow_from_directory(DATA_INPUT_PATH,
                                            target_size = TARGET_SIZE,
                                            batch_size = BATCH_SIZE,
                                            subset = "training",
                                            class_mode = "categorical")
    valid_generator = datagen.flow_from_directory(DATA_INPUT_PATH,
                                        target_size = TARGET_SIZE,
                                        batch_size = BATCH_SIZE,
                                        subset = "validation",
                                        class_mode = 'categorical')

    logger.info("Creating model")
    model = get_model((IMG_SIZE, IMG_SIZE, NUM_COLOR_BITS), NUM_CLASSES)
    model.compile(loss = "categorical_crossentropy", optimizer = 'Adam', metrics=['accuracy'])

    logger.info("Training model")
    callbacks = [
        stopTrainingCallback(),
        tf.keras.callbacks.ModelCheckpoint(
            "vgg16_v1_{epoch:02d}_{val_accuracy:.3f}.h5",
            monitor="val_accuracy",
            save_best_only=True,
            mode='max'
        )
    ]
    history = model.fit(train_generator,
                        epochs = EPOCHS,
                        steps_per_epoch = train_generator.samples//BATCH_SIZE,
                        validation_data = valid_generator,
                        validation_steps = valid_generator.samples//BATCH_SIZE,
                        callbacks = callbacks,
                        verbose = 1)
    logger.info(f"Training model accuracy: {history.history['accuracy'][-1]} val_accuracy: {history.history['val_accuracy'][-1]}")
    
    logger.info("Saving model")
    model.save(MODEL_FILE_PATH)

if __name__ == "__main__":
    main()