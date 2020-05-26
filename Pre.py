from PIL import Image
import os
import math
import numpy as np
import matplotlib.pyplot as plt


def shuffle_data(path, n, low_len=256):
    class_list = os.listdir(path)
    for a in range(n+1, n-1, -1):
        for class_name in class_list:
            image_list = os.listdir('{}/{}'.format(path, class_name))
            choices = np.random.permutation(len(image_list))
            i = 0
            for image_name in image_list:
                try:
                    image = Image.open('{}/{}/{}'.format(
                        path, class_name, image_name)).convert('RGB')
                except OSError:
                    print('{} of class {} could not open.'.format(image_name, class_name))
                    continue
                """
                w, h = image.size
                if w <= h:
                    ratio = low_len / w
                    image = image.resize((low_len, math.ceil(h * ratio)))
                else:
                    ratio = low_len / h
                    image = image.resize((math.ceil(w * ratio), low_len))
                """
                os.remove('{}/{}/{}'.format(path, class_name, image_name))
                image.save('{0:s}/{1:s}/{2:0{3}d}.jpg'.format(
                    path, class_name, choices[i], a))
                i += 1

    print('shuffling completed')
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


def load_image(path):
    """
    loading single image and crop size of it to (256, 192)
    :param path: image location
    :return: pillow Image
    """
    try:
        image = Image.open(path).convert('RGB')
        image = crop(image, 256, 192)
        return image
    except OSError:
        print('{} could not open.'.format(path))
        return None


def crop(image, width, height):
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
        return crop(image, width, height)

    w, h = image.size
    left = (w - width) / 2
    top = (h - height) / 2
    right = (w + width) / 2
    bottom = (h + height) / 2
    image = image.crop((left, top, right, bottom))
    return np.asarray(image)


if __name__ == '__main__':
    shuffle_data('Dataset', 4)

