""" Convolutional Neural Network Implemented by Minpy.model_builder """

from minpy.nn.solver import Solver
import minpy.nn.model_builder as builder

def main():
    # Specify network configurations.
    CNN = builder.Sequential(
        builder.Convolution((7, 7), 32),
        builder.ReLU(),
        builder.Pooling('max', (2, 2), (2, 2)),
        builder.Flatten(),
        builder.Affine(512),
        builder.ReLU(),
        builder.Affine(10)
    )

    # Cast the configuration to a model compatible with minpy solver.
    model = builder.Model(CNN, 'softmax', (3, 32, 32))

    # The above line is equivalent to using the ConvolutionNet, as below(uncomment).
    # model = TwoLayerNet()

    solver = Solver(
        model,
        # ...
    )

    # Training procedures are identical to previous examples.
