import numpy
from PIL import Image
import os
from keras.applications import MobileNet, VGG16
from keras.layers import Input, MaxPool2D
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input,  GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

global_path = os.getcwd()

train_dir = 'training'
test_dir = 'validation'
my_train = 'my_train'
IMG_SIZE = 160
num_classes = 10
batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    zoom_range=0.1,
    # preprocessing_function=brightness_adjustment
    )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


my_train_gen = test_datagen.flow_from_directory(my_train,
                                                target_size=(IMG_SIZE, IMG_SIZE),
                                                color_mode='rgb',  # mages will be resized to 150x150
                                                batch_size=batch_size,
                                                # ve_to_dir='results',
                                                # subset='training',
                                                class_mode='categorical'
                                                )
# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    train_dir,  # this is the target directory
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',  # mages will be resized to 150x150
    batch_size=batch_size,
    # ve_to_dir='results',
    # subset='training',
    class_mode='categorical')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    batch_size=batch_size,
    # subset='validation',
    class_mode='categorical')

def build_model():

    base_model = MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256)(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    #x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    preds = Dense(10,activation='softmax')(x) #final layer with softmax activation
    model=Model(inputs=base_model.input, outputs=preds)
    for layer in model.layers[:84]: # 87 mobilennet
        layer.trainable=False
    return model


if __name__ == "__main__":
    model = build_model()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)
    callbacks = [reduce_lr]
    history = model.fit_generator(train_generator, epochs=6, workers=1, callbacks=callbacks,
                                           steps_per_epoch=len(train_generator),
                                           validation_data=validation_generator,
                                           validation_steps=len(validation_generator),
                                           shuffle=True)
    model_json = model.to_json()
    with open(os.path.join(os.getcwd(),"model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(os.getcwd(),"model.h5"))
    print("Saved model to disk")
