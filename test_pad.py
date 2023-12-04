from gym_snake.envs.snake.controller import extract_and_pad
import numpy as np

import matplotlib.pyplot as plt

image = 255*np.ones((30, 30, 3), dtype=np.uint8)

image2 = extract_and_pad(image, -10, 31, -1, 45)

plt.imshow(image)
plt.show()
plt.imshow(image2)
plt.show()