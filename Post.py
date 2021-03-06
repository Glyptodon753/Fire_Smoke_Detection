from keras.models import load_model
from DataProcessing import Data
from PIL import Image
from keras import models
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import math


def confusion_matrix(model, dataset, indexes, classes=('Fire', 'Neutral', 'Smoke')):
    try:
        os.mkdir('confusion matrix')
    except FileExistsError:
        pass

    for i in range(len(classes)):
        for j in range(len(classes)):
            path = 'confusion matrix/{}_{}'.format(classes[i], classes[j])
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass
            os.mkdir(path)

    matrix = np.zeros((len(classes), len(classes)), dtype='int32')
    for class_ in range(len(classes)):
        for i in indexes:
            image = dataset.load_single_image(class_, i)
            predict = model.predict(np.expand_dims(
                image.astype('float32') / 255.0, axis=0))
            predict_label = int(np.argmax(predict))
            matrix[class_][predict_label] += 1
            image = Image.fromarray(image)
            image.save('confusion matrix/{0:s}_{1:s}/{2:s} {3:04d}.jpg'.format(
                classes[class_], classes[predict_label], classes[class_], i))

    return matrix


def plot_confusion_matrix(matrix, rate=True, classes=('Fire', 'Neutral', 'Smoke')):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.array(matrix), cmap='Oranges')

    width, height = matrix.shape
    for x in range(width):
        for y in range(height):
            if rate:
                s = '{0:.3f}'.format(matrix[x][y])
            else:
                s = str(matrix[x][y])
            ax.annotate(s, xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    ax.xaxis.tick_top()
    if rate:
        plt.savefig('Chart/confusion_matrix_rate.png', format='png')
    else:
        plt.savefig('Chart/confusion_matrix.png', format='png')
    plt.show()


def plot_accuracy_loss(history, plot=True):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('Chart/accuracy.png', format='png')
    if plot:
        plt.show()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('Chart/loss.png', format='png')
    if plot:
        plt.show()


def feature_map(model, x, classes=('Fire', 'Neutral', 'Smoke')):
    """
    Getting feature maps of last convolution layer of image from given model.
    :param model: keras model, trained model.
    :param x: numpy array, raw image to predict
    :param classes: string tuple, names of classes in model
    :return: None
    """
    x = np.expand_dims(x, axis=0)
    predict = model.predict(x)
    class_idx = np.argmax(predict[0])
    print(predict)
    print(classes[int(class_idx)])

    # Finding last convolution layer.
    layer = 0
    i = 0
    for s in model.layers:
        if str(s).find('conv') >= 0:
            layer = i
        i += 1

    layer_outputs = [layer.output for layer in model.layers[:layer+1]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x)

    plt.rcParams['font.size'] = 5
    fig = plt.figure()

    feature = activations[layer]
    maps = feature.shape[-1]
    title = [_ for _ in range(maps)]
    square = math.ceil(math.sqrt(maps))
    for i in range(maps):
        plt.subplot(square, square, i+1)
        plt.imshow(feature[0, :, :, i])
        plt.title(title[i])
        plt.axis('off')

    fig.tight_layout()
    plt.savefig('feature map.png')
    # plt.show()


if __name__ == '__main__':
    model = load_model('FS.h5')
    classes = ('Fire', 'Neutral', 'Smoke')
    dataset = Data('Dataset', 1600, classes)
    cm = confusion_matrix(model, dataset, dataset.validation_set, classes)
    plot_confusion_matrix(cm, False, classes)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm, True, classes)
    test_loss, test_acc = model.evaluate_generator(dataset.generator('test', batch_size=5),
                                                   steps=dataset.get_size('test')//5,
                                                   verbose=1)
    print('test accuracy: {}'.format(test_acc))
    print('test loss: {}'.format(test_loss))
