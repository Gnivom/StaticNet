# Static Neural Network

## What is this?

It's a strongly typed neural network. Models' architectures are decided at compile time. If this seems like an unnecessary hassle, it's because it is. Tell me if you find a use case!

## What does it do?

You can assemble networks by layering dense or convolutional layers with a selection of activation functions. You can optimize them with a selection of gradient descent backpropagation algorithms, with respect to a selection of loss functions.

## How do I use it?

Check `examples/mnist/`

## Dependencies

- CMake 3.10 or later
- C++17-compliant compiler

## Setup

```
$ mkdir build
$ cd build
$ cmake ..
```
