from itertools import chain
import numpy
import scipy.signal
import mxnet as mx
import minpy.nn.init
import minpy.core as core
import minpy.numpy as np
from minpy.nn.model import ModelBase

class Agent(ModelBase):
    def __init__(self, input_size, act_space, config):
        super(Agent, self).__init__()
        self.ctx = config.ctx
        self.act_space = act_space
        self.config = config
        self.add_param('fc1', (config.hidden_size, input_size))
        self.add_param('policy_fc_last', (act_space, config.hidden_size))
        self.add_param('vf_fc_last', (1, config.hidden_size))
        self.add_param('vf_fc_last_bias', (1,))

        self._init_params()

        self.optim_configs = {}
        for p in self.param_configs:
            self.optim_configs[p] = {'learning_rate': self.config.learning_rate}

    def forward(self, X):
        a = np.dot(self.params['fc1'], X.T)
        h = np.maximum(0, a)
        logits = np.dot(h.T, self.params['policy_fc_last'].T)
        ps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        ps /= np.sum(ps, axis=1, keepdims=True)
        vs = np.dot(h.T, self.params['vf_fc_last'].T) + self.params['vf_fc_last_bias']
        return ps, vs

    def loss(self, ps, as_, vs, rs, advs):
        ps = np.maximum(1.0e-5, np.minimum(1.0 - 1e-5, ps))
        policy_grad_loss = -np.sum(np.log(ps) * as_ * advs)
        vf_loss = 0.5*np.sum((vs - rs)**2)
        entropy = -np.sum(ps*np.log(ps))
        loss_ = policy_grad_loss + self.config.vf_wt*vf_loss - self.config.entropy_wt*entropy
        return loss_

    def act(self, ps):
        us = numpy.random.uniform(size=ps.shape[0])[:, np.newaxis]
        as_ = (numpy.cumsum(ps.asnumpy(), axis=1) > us).argmax(axis=1)
        return as_

    def train_step(self, env_xs, env_as, env_rs, env_vs):
        # Stack all the observations and actions.
        xs = np.vstack(list(chain.from_iterable(env_xs)))
        as_ = numpy.array(list(chain.from_iterable(env_as)))[:, np.newaxis]
        # One-hot encode the actions.
        buf = mx.nd.array(numpy.zeros([xs.shape[0], self.act_space]), self.ctx)
        as_ = mx.nd.onehot_encode(mx.nd.array(as_.ravel(), self.ctx), buf).asnumpy()

        # Compute discounted rewards and advantages.
        drs, advs = [], []
        gamma, lambda_ = self.config.gamma, self.config.lambda_
        for i in xrange(len(env_vs)):
            # Compute discounted rewards with a 'bootstrapped' final value.
            rs_bootstrap = [] if env_rs[i] == [] else env_rs[i] + [env_vs[i][-1]]
            drs.extend(self._discount(rs_bootstrap, gamma)[:-1])

            # Compute advantages using Generalized Advantage Estimation;
            # see eqn. (16) of [Schulman 2016].
            delta_t = env_rs[i] + gamma*numpy.array(env_vs[i][1:]) - numpy.array(env_vs[i][:-1])
            advs.extend(self._discount(delta_t, gamma * lambda_))

        drs = numpy.array(drs)[:, np.newaxis]
        advs = numpy.array(advs)[:, np.newaxis]

        def loss_func(*params):
            ps, vs = self.forward(xs)
            loss_ = self.loss(ps, as_, vs, drs, advs)
            return loss_

        grads = self._forward_backward(loss_func)
        self._update_params(grads)

    def _discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def _forward_backward(self, loss_func):
        param_arrays = list(self.params.values())
        param_keys = list(self.params.keys())
        grad_and_loss_func = core.grad_and_loss(loss_func, argnum=range(len(param_arrays)))
        grad_arrays, loss = grad_and_loss_func(*param_arrays)
        grads = dict(zip(param_keys, grad_arrays))
        if self.config.grad_clip:
            for k, v in grads.iteritems():
                grads[k] = numpy.clip(v, -self.config.clip_magnitude, self.config.clip_magnitude)

        return grads

    def _update_params(self, grads):
        for p, w in self.params.iteritems():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.config.update_rule(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config

    def _init_params(self):
        for name, config in self.param_configs.items():
            init_func = minpy.nn.init.constant if name.endswith('bias') else self.config.init_func
            self.params[name] = init_func(config['shape'], self.config.init_config)
