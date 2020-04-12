from keras.models import load_model
from DataProcessing import Data
import keras.backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt

dataset = Data('Dataset', 1200)

a = 0
n = 150
classes = ('Fire', 'Neutral', 'Smoke')
class_name = classes[a]
x = dataset.load_single_image(a, n).astype('float32') / 255.0
x = np.expand_dims(x, axis=0)

model = load_model('FS.h5')
last_conv_layer = model.get_layer('conv2d_3')
pred = model.predict(x)
class_idx = np.argmax(pred[0])
print(pred)

class_output = model.output[:, class_idx]
gap_weights = model.get_layer("global_average_pooling2d_1")

grads = K.gradients(class_output, gap_weights.output)[0]
iterate = K.function([model.input], [grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
print(pooled_grads_value.shape)
pooled_grads_value = np.squeeze(pooled_grads_value, axis=0)
print(pooled_grads_value.shape)
for i in range(len(pooled_grads_value)):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

s = 'confusion matrix/{0:s}_{1:s}/{2:s} {3:04d}.jpg'.format(
    class_name, classes[int(class_idx)], class_name, n)
print(s)
img = cv2.imread(s)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
plt.matshow(heatmap)
plt.show()
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
print(type(img[0][0][0]))
superimposed_img = heatmap * 0.4 + img

cv2.imwrite('cam.jpg', superimposed_img)




