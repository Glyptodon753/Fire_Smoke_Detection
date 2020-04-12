from keras.models import load_model
from DataProcessing import Data
import keras.backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt


K.clear_session()

dataset = Data('Dataset', 1200)
n = 11
x = dataset.load_single_image(0, n).astype('float32') / 255.0
x = np.expand_dims(x, axis=0)

model = load_model('FS.h5')
fire_output = model.output[:, 2]
last_conv_layer = model.get_layer('conv2d_3')
print(model.predict(x))

grads = K.gradients(fire_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])
print(pooled_grads_value)

for i in range(pooled_grads_value.shape[0]):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

s = 'confusion matrix/Smoke_Smoke/Smoke {0:04d}.jpg'.format(n)
print(s)
img = cv2.imread(s)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

cv2.imwrite('cam.jpg', superimposed_img)


