from PIL import Image
import matplotlib.pyplot as plt
import os


files = os.listdir('sky')
for file in files:
    image_file = Image.open("sky/{}".format(file))
    image_file = image_file.convert('L')
    plt.matshow(image_file, cmap='gray')
    plt.show()

