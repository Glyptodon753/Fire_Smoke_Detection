from keras.models import load_model
from DataProcessing import Data
from Post import feature_map
from Pre import load_image
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2


def cam(model, x, threshold=0.3, layer=3, classes=('Fire', 'Neutral', 'Smoke')):
    """
    Class Activation Map for image
    :param model: keras model, trained model
    :param x: numpy array, raw image to predict
    :param threshold: float, threshold to extract area activated by class
    :param layer: integer, layer-th convolution layer
    :param classes: string tuple, names of classes in model
    :return: class which model predict and class area
    """
    x = np.expand_dims(x, axis=0)

    last_conv_layer = model.get_layer('conv2d_{}'.format(layer))
    predict = model.predict(x)
    class_idx = np.argmax(predict[0])
    print(predict)
    print(class_idx)
    print(classes[int(class_idx)])

    class_output = model.output[:, class_idx]
    gap_weights = model.get_layer("global_average_pooling2d_1")

    grads = K.gradients(class_output, gap_weights.output)[0]
    iterate = K.function([model.input], [grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    pooled_grads_value = np.squeeze(pooled_grads_value, axis=0)
    # print(pooled_grads_value)
    for i in range(len(pooled_grads_value)):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    # plt.matshow(heatmap)
    # plt.show()
    heatmap = cv2.resize(heatmap, (x.shape[2], x.shape[1]))

    class_area = np.sum(heatmap > threshold)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    x = np.squeeze(x, axis=0) * 255.0
    x = x.astype('uint8')[:, :, ::-1]  # BGR to RGB

    superimposed_img = heatmap * 0.4 + x

    combine = np.concatenate((x, heatmap, superimposed_img), axis=1)
    cv2.imwrite('origin_heatmap_cam.png', combine)

    cv2.imwrite('Class Activation Map/origin.jpg', x)
    cv2.imwrite('Class Activation Map/heatmap.jpg', heatmap)
    cv2.imwrite('Class Activation Map/cam.jpg', superimposed_img)

    return classes[int(class_idx)], class_area


if __name__ == '__main__':
    classes = ('Fire', 'Neutral', 'Smoke')
    dataset = Data('Dataset', 160, classes)
    image = dataset.load_single_image(2, 19).astype('float32') / 255.0
    model = load_model('FS.h5')
    predict_class, area = cam(model, image, 0.3, 3, classes)
    image_area = image.shape[0] * image.shape[1]
    print('{0:s}: {1:d}/{2:d} pixels'.format(predict_class, area, image_area))
    print('{0:s}: {1:.5f} %'.format(predict_class, area/image_area))

    feature_map(model, image, classes)

