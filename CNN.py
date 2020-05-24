from keras import layers
from keras import models
from Post import plot_accuracy_loss, plot_confusion_matrix, confusion_matrix
from DataProcessing import Data
import numpy as np


def runCNN(input_shape, filters, dense, output_shape, epochs):
    model = models.Sequential()
    model.add(layers.Conv2D(filters[0], (3, 3),
                            activation='relu', input_shape=input_shape))
    for f in filters[1:]:
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(f, (3, 3),
                                activation='relu', input_shape=input_shape))

    model.add(layers.GlobalAveragePooling2D(input_shape=input_shape))

    for d in dense:
        model.add(layers.Dense(d, activation='relu'))
    model.add(layers.Dense(output_shape, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    dataset = Data('Feature Map', 1600)
    train_size = dataset.get_size('train')
    validation_size = dataset.get_size('validation')
    batch = 8

    record = model.fit_generator(dataset.generator('train', batch_size=batch),
                                 steps_per_epoch=train_size//batch,
                                 epochs=epochs,
                                 validation_data=dataset.generator('validation', batch_size=batch),
                                 validation_steps=validation_size//batch,
                                 verbose=1)
    model.save('FS.h5')

    plot_accuracy_loss(record.history)

    cm = confusion_matrix(model, dataset, dataset.validation_set)
    plot_confusion_matrix(cm, False)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm, True)
    test_loss, test_acc = model.evaluate_generator(
        dataset.generator('test', batch_size=5),
        steps=dataset.get_size('test') // 5, verbose=1)

    with open('information.txt', 'w') as fp:
        fp.write('input shape: {}\n'.format(input_shape))
        fp.write('filters: {}\n'.format(filters))
        fp.write('dense: {}\n'.format(dense))
        fp.write('output shape: {}\n'.format(output_shape))
        fp.write('epochs: {}\n'.format(epochs))
        fp.write('test accuracy: {}\n'.format(test_acc))
        fp.write('test loss: {}\n'.format(test_loss))

    print('test accuracy: {}'.format(test_acc))
    print('test loss: {}'.format(test_loss))

