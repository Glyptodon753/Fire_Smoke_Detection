from PIL import Image
import numpy as np
import math


class Data:
    def __init__(self,
                 path,
                 class_n,
                 classes=('Fire', 'Neutral', 'Smoke'),
                 test_rate=0.2,
                 validation_rate=0.1,
                 ):
        self.path = path
        self.classes = classes
        self.test_set = np.arange(0, int(class_n * test_rate))
        self.validation_set = np.arange(int(class_n * test_rate), int(class_n * (test_rate + validation_rate)))
        self.train_set = np.arange(int(class_n * (test_rate + validation_rate)), class_n)
        self.test_labels = np.zeros((len(self.test_set)*3,)).astype('int32')
        self.test_images = np.zeros((len(self.test_set)*3,)).astype('int32')

    def load_single_image(self, class_, index):
        try:
            image = Image.open('{0:s}/{1:s}/{2:04d}.jpg'.format(
                self.path, self.classes[class_], index)).convert('RGB')
            image = self.crop(image, 256, 192)
            return image
        except OSError:
            print('{} of class {} could not open.'.format(index, self.classes[class_]))
            return None

    def crop(self, image, width, height):
        w, h = image.size

        ww = w / width
        hh = h / height
        if ww <= hh:
            ratio = width / w
            image = image.resize((width, math.ceil(h * ratio)))
        else:
            ratio = height / h
            image = image.resize((math.ceil(w * ratio), height))

        if image.size[0] < width or image.size[1] < height:
            return self.crop(image, width, height)

        w, h = image.size
        left = (w - width) / 2
        top = (h - height) / 2
        right = (w + width) / 2
        bottom = (h + height) / 2
        image = image.crop((left, top, right, bottom))
        return np.asarray(image)

    def generator(self, what_set='train', batch_size=64):
        if what_set == 'train':
            dataset = self.train_set
        elif what_set == 'validation':
            dataset = self.validation_set
        else:
            dataset = self.test_set
            count = 0
            for class_ in range(len(self.classes)):
                self.test_labels[count:count+len(self.test_set)] = class_
                self.test_images[count:count+len(self.test_set)] = self.test_set
                count += len(self.test_set)
        count = 0
        while True:
            images = []
            labels = []

            if what_set == 'test':
                label_ = self.test_labels[count:count+batch_size]
                image_ = self.test_images[count:count+batch_size]
                count += batch_size
            else:
                label_ = np.random.choice(len(self.classes), batch_size, replace=True)
                image_ = np.random.choice(dataset, batch_size, replace=False)

            for i in range(batch_size):
                label = np.zeros((len(self.classes),))
                label[label_[i]] = 1
                image = self.load_single_image(label_[i], image_[i]).astype('float32') / 255.0

                labels.append(label)
                images.append(image)

            yield np.asarray(images), np.asarray(labels)

    def get_size(self, what_set='train'):
        if what_set == 'train':
            return len(self.classes) * len(self.train_set)
        elif what_set == 'validation':
            return len(self.classes) * len(self.validation_set)
        else:
            return len(self.classes) * len(self.test_set)

    def get_test(self):
        return self.test_images, self.test_labels
