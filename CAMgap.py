from keras.models import load_model
from DataProcessing import Data
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2


def cam(model, x, threshold=0.3, classes=('Fire', 'Neutral', 'Smoke')):
    x = np.expand_dims(x, axis=0)

    last_conv_layer = model.get_layer('conv2d_3')
    predict = model.predict(x)
    class_idx = np.argmax(predict[0])
    print(predict)
    print(classes[int(class_idx)])

    class_output = model.output[:, class_idx]
    gap_weights = model.get_layer("global_average_pooling2d_1")

    grads = K.gradients(class_output, gap_weights.output)[0]
    iterate = K.function([model.input], [grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    pooled_grads_value = np.squeeze(pooled_grads_value, axis=0)
    print(pooled_grads_value)
    # for i in range(len(pooled_grads_value)):
    #    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.show()

    heatmap = cv2.resize(heatmap, (x.shape[2], x.shape[1]))
    class_area = np.sum(heatmap > threshold)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    x = np.squeeze(x, axis=0) * 255.0
    x = x.astype('uint32')[:, :, ::-1]
    superimposed_img = heatmap * 0.4 + x

    cv2.imwrite('cam.jpg', superimposed_img)

    return class_area


dataset = Data('Dataset', 1200)
image = dataset.load_single_image(0, 999).astype('float32') / 255.0

model = load_model('FS.h5')
area = cam(model, image)
print(area)
print(area/(image.shape[0]*image.shape[1]))
