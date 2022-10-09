import torch
from torchvision import datasets, transforms
import torchvision.datasets.mnist as mnist
import os

def main():
    data_train = datasets.MNIST(root = "./raw_mnist/",transform=transforms.ToTensor(),train = True,download = True)
    root = './raw_mnist/MNIST/raw'
    train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )

    torch.save(train_set,'./raw_mnist/mnist_tensor.pt')

if __name__ == '__main__':
    main()