from keras.models import load_model
from DataProcessing import Data
from PIL import Image
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


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


def plot_confusion_matrix(matrix, classes=('Fire', 'Neutral', 'Smoke')):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.array(matrix), cmap='Oranges')

    width, height = matrix.shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(matrix[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    ax.xaxis.tick_top()
    plt.savefig('Chart/confusion_matrix.png', format='png')
    plt.show()


def plot_accuracy_loss(history):
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
    plt.show()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('Chart/loss.png', format='png')
    plt.show()


if __name__ == '__main__':
    model = load_model('FS.h5')
    dataset = Data('Dataset', 1200)
    cm = confusion_matrix(model, dataset, dataset.validation_set)

    plot_confusion_matrix(cm)
"""
    test_loss, test_acc = model.evaluate_generator(dataset.generator('test', batch_size=10),
                                                   steps=dataset.get_size('test')//10,
                                                   verbose=1)
    print('test accuracy: {}'.format(test_acc))
    print('test loss: {}'.format(test_loss))

"""