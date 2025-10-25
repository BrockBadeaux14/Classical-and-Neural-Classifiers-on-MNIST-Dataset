import numpy as np
import os
from PIL import Image
import glob

mnist_path = '../MNIST'

def load_mnist_data():
    """
    Load MNIST data from the folder structure
    Returns:
        images, labels    
    """
    images = []
    labels = []
    
    for digit in range(10):
        digit_folder = os.path.join(mnist_path, str(digit))
        if os.path.exists(digit_folder):
            image_files = glob.glob(os.path.join(digit_folder, '*.png'))
            
            for image_file in image_files:
                img = Image.open(image_file).convert('L')
                img_array = np.array(img)
                
                images.append(img_array)
                labels.append(digit)
    
    return np.array(images), np.array(labels)