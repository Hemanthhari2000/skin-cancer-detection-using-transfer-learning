from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

TRAIN_PATH = 'dataset/train/'
VALIDATION_PATH = 'dataset/test/'

# PREDICT_IMAGE_PATH = 'dataset/predictions/bad_image.jpg'


def create_model():

    VGG = VGG16(input_shape=(224, 224, 3),
                include_top=False, weights='imagenet')

    for layer in VGG.layers:
        layer.trainable = False

    x = Flatten()(VGG.output)
    prediction = Dense(2, activation='softmax')(x)

    model = Model(inputs=VGG.input, outputs=prediction)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def data_aug():

    train_datagen = ImageDataGenerator(rescale=1./255.,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0/255.)

    training_set = train_datagen.flow_from_directory(TRAIN_PATH,
                                                     batch_size=32,
                                                     target_size=(224, 224),
                                                     class_mode='categorical')

    test_set = test_datagen.flow_from_directory(VALIDATION_PATH,
                                                batch_size=32,
                                                target_size=(224, 224),
                                                class_mode='categorical')

    return training_set, test_set


def fit_and_save(model, training_set, test_set, EPOCHS=10):
    history = model.fit_generator(training_set,
                                  validation_data=test_set,
                                  epochs=EPOCHS,
                                  steps_per_epoch=len(training_set),
                                  validation_steps=len(test_set))
    model.save('sav/vggmodel.h5')
    return history


def visualize(history):
    # Losses
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()

    # accuracies
    # plt.plot(history.history['acc'], label='train acc')
    # plt.plot(history.history['val_acc'], label='val acc')
    # plt.legend()
    # plt.show()


def predict_the_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = np.asarray(img)
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    saved_model = load_model('sav/vggmodel.h5')
    output = saved_model.predict(img)[0]
    if output[0] > output[1]:
        print(f'Benign with probability:\t{round(output[0], 2)}')
    else:
        print(f'Malignant with probability:\t{round(output[1], 2)}')


def main():
    model = create_model()
    training_set, test_set = data_aug()
    print(model.summary())
    history = fit_and_save(model, training_set, test_set, EPOCHS=5)
    visualize(history)
    predict_the_image(PREDICT_IMAGE_PATH)


if __name__ == "__main__":
    main()
