from PIL import Image
import os
import math
import numpy as np
import matplotlib.pyplot as plt


def shuffle_data(path):
    class_list = os.listdir(path)
    low_len = 256

    for class_name in class_list:
        image_list = os.listdir('{}/{}'.format(path, class_name))
        choices = np.random.permutation(len(image_list))
        i = 0
        for image_name in image_list:
            try:
                image = Image.open('{}/{}/{}'.format(path, class_name, image_name)).convert('RGB')
            except OSError:
                print('{} of class {} could not open.'.format(image_name, class_name))
                continue
            w, h = image.size
            if w <= h:
                ratio = low_len / w
                image = image.resize((low_len, math.ceil(h * ratio)))
            else:
                ratio = low_len / h
                image = image.resize((math.ceil(w * ratio), low_len))

            os.remove('{}/{}/{}'.format(path, class_name, image_name))
            image.save('{0:s}/{1:s}/{2:04d}.jpg'.format(path, class_name, choices[i]))
            i += 1

        print('class {} completed.'.format(class_name))

    print('all classes completed')
    return None


def images_to_npz(path, npz_name, width=256, height=256):
    class_list = os.listdir(path)
    data = {}

    for class_name in class_list:
        image_list = os.listdir('{}/{}'.format(path, class_name))
        image_list.sort()
        tmp_array = np.empty((len(image_list), height, width, 3), dtype='uint8')
        i = 0
        for image_name in image_list:
            try:
                image = Image.open('{}/{}/{}'.format(path, class_name, image_name)).convert('RGB')
            except OSError:
                print('{} of class {} could not open.'.format(image_name, class_name))
                continue
            w, h = image.size
            if w <= h:
                ratio = width / w
                image = image.resize((width, math.ceil(h*ratio)))
            else:
                ratio = height / h
                image = image.resize((math.ceil(w*ratio), height))

            w, h = image.size
            left = (w - width) / 2
            top = (h - height) / 2
            right = (w + width) / 2
            bottom = (h + height) / 2
            image = image.crop((left, top, right, bottom))
            tmp_array[i] = np.asarray(image)
            i += 1

        print('class {} completed.'.format(class_name))
        data[class_name] = tmp_array

    np.savez_compressed(npz_name, **data)
    print('all classes completed')
    return None


def sub_plot(images, title, width, height):
    plt.figure()
    position = 1
    for i in range(height):
        for j in range(width):
            plt.subplot(height, width, position)
            plt.imshow(images[position - 1])
            plt.title(title[position-1])
            plt.axis('off')
            position += 1
    plt.show()
    return None


if __name__ == '__main__':
    shuffle_data('Dataset')

