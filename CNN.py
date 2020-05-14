from keras import layers
from keras import models
from Post import plot_accuracy_loss
from DataProcessing import Data


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='tanh', input_shape=(192, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='tanh'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='tanh'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(256, activation='tanh'))
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

dataset = Data('Dataset', 1600)
train_size = dataset.get_size('train')
validation_size = dataset.get_size('validation')
batch = 8

record = model.fit_generator(dataset.generator('train', batch_size=batch),
                             steps_per_epoch=train_size//batch,
                             epochs=300,
                             validation_data=dataset.generator('validation', batch_size=batch),
                             validation_steps=validation_size//batch,
                             verbose=1)
model.save('FS.h5')

plot_accuracy_loss(record.history)

