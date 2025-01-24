

import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
# Download training dataset
dataset = MNIST(root='data/', download=True)

# Import the summary writer
from torch.utils.tensorboard import SummaryWriter# Create an instance of the object
writer = SummaryWriter()

# Compose a set of transforms to use later on
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
#
# datasets.MNIST.resources = [
#     ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
#     ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
#     ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
#     ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
# ]
# new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
# datasets.MNIST.resources = [
#    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
#    for url, md5 in datasets.MNIST.resources
# ]



# Load in the MNIST dataset
trainset = datasets.MNIST(
    'mnist_train',
    train=True,
    download=True,
    transform=transform
)

# Create a data loader
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

# Get a pre-trained ResNet18 model
model = torchvision.models.resnet18(False)
# Change the first layer to accept grayscale images
# 修改第一层，接受灰度图
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Get the first batch from the data loader
images, labels = next(iter(trainloader))

# Write the data to TensorBoard
grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)
writer.close()
