using Flux, MLDatasets, CUDA
using Flux: train!, onehotbatch

# Load training data (images, labels)
x_train, y_train = MLDatasets.MNIST.traindata()
# Load test data (images, labels)
x_test, y_test = MLDatasets.MNIST.testdata()
# Convert grayscale to float
x_train = Float32.(x_train)
# Create labels batch
y_train = Flux.onehotbatch(y_train, 0:9)