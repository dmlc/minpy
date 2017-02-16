"""Trains an agent to play Pong using policy gradients using the `RLPolicyGradientSolver`.
A reinforcement learning example in MinPy based on Karpathy's 'Pong from Pixels'.
    Ref: http://karpathy.github.io/2016/05/31/rl/
"""

import argparse

import gym
import minpy.numpy as np
from minpy.nn.model import ModelBase
import numpy

from rl_policy_gradient_solver import RLPolicyGradientSolver

class CartPolePreprocessor(object):
    def __init__(self):
        self.prev = None

    def reset(self):
        self.prev = None

    def preprocess(self, state):
        """ Preprocess a 4 float vector."""
        return state


class PolicyNetwork(ModelBase):
    def __init__(self,
                 preprocessor,
                 input_size=4,
                 hidden_size=128,
                 gamma=0.99):  # Reward discounting factor
        super(PolicyNetwork, self).__init__()
        self.preprocessor = preprocessor
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.add_param('w1', (hidden_size, input_size))
        self.add_param('w2', (2, hidden_size))

    def forward(self, X):
        """Forward pass to obtain the action probabilities for each observation in `X`."""
        a = np.dot(self.params['w1'], X.T)
        h = np.maximum(0, a)
        logits = np.dot(h.T, self.params['w2'].T)
        return logits

    def choose_action(self, logits):
        """Return using the probability float `p`."""
        e_logits = np.exp(logits - np.max(logits))
        p = e_logits / e_logits.sum()
        a = 0 if numpy.random.uniform() < p else 1
        y = 1 if a == 1 else 0
        return a, y

    def loss(self, ps, ys, rs):
        # Prevent log of zero.
        N = ps
        C = ps
        if len(ys.shape) == 1:
            #convert it to one hot encoding
            onehot_label = np.zeros([N, C])
            np.onehot_encode(ys, onehot_label)
        else:
            onehot_label = ys
        prob = np.softmax_output(ps, onehot_label)
        return -np.sum(np.log(prob) * onehot_label * rs)

    def discount_rewards(self, rs):
        drs = np.zeros_like(rs).asnumpy()
        s = 0
        for t in reversed(xrange(0, len(rs))):
            # Reset the running sum at a game boundary.
            if rs[t] != 0:
                s = 0
            s = s * self.gamma + rs[t]
            drs[t] = s
        drs -= np.mean(drs)
        drs /= np.std(drs)
        return drs


def main(args):
    if args.gpu:
        from minpy.context import set_context, gpu
        set_context(gpu(0))  # set the global context as gpu(0)

    env = gym.make("Pong-v0")
    env.seed(args.seed)
    numpy.random.seed(args.seed)

    model = PolicyNetwork(PongPreprocessor())
    solver = RLPolicyGradientSolver(model, env,
                                    update_rule='rmsprop',
                                    optim_config={
                                        'learning_rate': args.learning_rate,
                                        'decay_rate': args.decay_rate
                                    },
                                    init_rule='custom',
                                    init_config={
                                        'function': lambda shape: np.random.randn(shape[0], shape[1]) / numpy.sqrt(shape[1])
                                    },
                                    render=args.render,
                                    save_dir=args.save_dir,
                                    save_every=args.save_every,
                                    resume_from=args.resume_from,
                                    num_episodes=args.num_episodes,
                                    verbose=args.verbose,
                                    print_every=args.print_every)
    solver.init()
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save-dir', default='./')
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--resume-from', default=None)
    parser.add_argument('--num-episodes', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--decay-rate', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    print('args=%s' % args)
    main(args)
