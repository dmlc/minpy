import numpy as np
import minpy.nn.init
import minpy.nn.optim
import mxnet as mx

class Config(object):
    def __init__(self, args):
        # Default training settings
        self.ctx = mx.gpu(0) if args.gpu else mx.cpu()
        self.init_func = minpy.nn.init.custom
        self.init_config = {
            'function': lambda shape: np.random.randn(shape[0], shape[1]) / np.sqrt(shape[1])
        }
        self.learning_rate = 1e-3
        self.update_rule = minpy.nn.optim.rmsprop
        self.grad_clip = True
        self.clip_magnitude = 40.0

        # Default model settings
        self.hidden_size = 200
        self.gamma = 0.99
        self.lambda_ = 1.0
        self.vf_wt = 0.5        # Weight of value function term in the loss
        self.entropy_wt = 0.01  # Weight of entropy term in the loss

        # Override defaults with values from `args`.
        for arg in self.__dict__:
            if arg in args.__dict__:
                self.__setattr__(arg, args.__dict__[arg])
