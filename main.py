from CNN import runCNN
from DataProcessing import Data

dataset = Data('Dataset', 1600, ('Fire', 'Neutral', 'Smoke'))
input_shape = (192, 256, 3)
filters = (32, 64, 64)
dense = (512, )
output_shape = 3
epochs = 100
runCNN(dataset, input_shape, filters, dense, output_shape, epochs)
