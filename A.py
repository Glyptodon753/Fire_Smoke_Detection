from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np


path = 'Dataset/Smoke/0999.jpg'
image = Image.open(path).convert('RGB')
image = image.filter(ImageFilter.GaussianBlur(radius=2))
plt.figure()
plt.imshow(image)
plt.show()


