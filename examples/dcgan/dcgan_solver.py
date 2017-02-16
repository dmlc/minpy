import time
import pickle
import os

import minpy.numpy as np
from minpy import core
from minpy.nn.solver import Solver


class DCGanSolver(Solver):
    """A custom `Solver` for Discriminative Generative GAN models

    Specifically, the model should provide:
        .forward(X)
        .loss(xs, ys, rs)
    """

    def __init__(self, gnet, dnet, **kwargs):
        """ Construct a new `DCGanSolver` instance.

        Parameters
        ----------
        gnet : ModelBase
            Generative Network.
        dnet : ModelBase
            Discriminative Network.

        Other Parameters
        ----------------
        num_episodes : int, optional
            Number of episodes to train for.
        update_every : int, optional
            Update model parameters every `update_every` episodes.
        save_every : int, optional
            Save model parameters in the `save_dir` directory every `save_every` episodes.
        save_dir : str, optional
            Directory to save model parameters in.
        resume_from : str, optional
            Loads a parameter file at this location, resuming model training with those parameters.
        render : boolean, optional
            Render the game environment when `True`.
        """
        self.gnet = gnet
        self.dnet = dnet
        self.num_episodes = kwargs.pop('num_episodes', 100000)
        self.update_every = kwargs.pop('update_every', 10)
        self.save_every = kwargs.pop('save_every', 10)
        self.save_dir = kwargs.pop('save_dir', './')
        self.resume_from = kwargs.pop('resume_from', None)
        self.render = kwargs.pop('render', False)

        super(, self).__init__(gnet, None, None, **kwargs)

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self._reset_data_iterators()

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.gnet.param_configs:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d
        
        for p in self.dnet.param_configs:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d
        # Overwrite it if the model specify the rules

        # Make a deep copy of the init_config for each parameter
        # and set each param to their own init_rule and init_config
        self.init_rules = {}
        self.init_configs = {}
        for p in self.gnet.param_configs:
            if 'init_rule' in self.gnet.param_configs[p]:
                init_rule = self.gnet.param_configs[p]['init_rule']
                init_config = self.gnet.param_configs[p].get('init_config',
                                                              {})
            else:
                init_rule = self.init_rule
                init_config = {k: v for k, v in self.init_config.items()}
            # replace string name with actual function
            if not hasattr(init, init_rule):
                raise ValueError('Invalid init_rule "%s"' % init_rule)
            init_rule = getattr(init, init_rule)
            self.init_rules[p] = init_rule
            self.init_configs[p] = init_config

        for p in self.dnet.param_configs:
            if 'init_rule' in self.dnet.param_configs[p]:
                init_rule = self.dnet.param_configs[p]['init_rule']
                init_config = self.dnet.param_configs[p].get('init_config',
                                                              {})
            else:
                init_rule = self.init_rule
                init_config = {k: v for k, v in self.init_config.items()}
            # replace string name with actual function
            if not hasattr(init, init_rule):
                raise ValueError('Invalid init_rule "%s"' % init_rule)
            init_rule = getattr(init, init_rule)
            self.init_rules[p] = init_rule
            self.init_configs[p] = init_config

    def _step(self, real_batch, rand_batch):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        ##########################
        # (1) train DNet with real
        ##########################
        # Compute loss and gradient
        def dnet_loss_func(batch, *params): # pylint: disable=unused-argument
            """
            Loss function calculate the loss
            """

            # It seems that params are not used in forward function. But since we will pass
            # model.params as arguments, we are ok here.
            predict = self.dnet.forward_batch(batch, mode='train')
            return self.model.loss_batch(batch, predict)

        dnet_param_arrays = list(self.dnet.params.values())
        dnet_param_keys = list(self.dnet.params.keys())
        dnet_grad_and_loss_func = core.grad_and_loss(
            dnet_loss_func, argnum=range(len(param_arrays)))
        dnet_grad_arrays_real, dnet_loss_real = dnet_grad_and_loss_func(real_batch, *dnet_param_arrays)
        dnet_grads_real = dict(zip(dnet_param_keys_real, dnet_grad_arrays_real))

        ##########################
        # (2) train DNet with fake
        ##########################
        fake_batch = real_batch
        fake_batch.data[0] = self.gnet.forward_batch(rand_batch, mode='train')
        fake_batch.label[0] = np.zeros(np.shape(fake_batch.label[0]))
        dnet_grad_arrays_fake, dnet_loss_fake = dnet_grad_and_loss_func(fake_batch, *dnet_param_arrays)
        dnet_grads_fake = dict(zip(dnet_param_keys_fake, dnet_grad_arrays_fake))

        # Perform a parameter update for dnet
        for p, w in self.dnet.params.items():
            dw = dnet_grads_real[p] + dnet_grads_fake[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.dnet.params[p] = next_w
            self.optim_configs[p] = next_config

        ##########################
        # (3) train GNet
        ##########################
        # use fake data, but give real label to bp
        fake_batch = real_batch
        fake_batch.data[0] = self.gnet.forward_batch(rand_batch, mode='train')
        # TODO: Do not call ff once more, just bp
        dnet_grad_arrays_for_gnet, dnet_loss_for_gnet = dnet_grad_and_loss_func(fake_batch, *dnet_param_arrays)
        dnet_bottom_grad = dnet_grad_arrays_real['data']

        def gnet_loss_func(gnet_out, dnet_bottom_grad, *params): # pylint: disable=unused-argument
            """
            Loss function calculate the loss
            """

            # It seems that params are not used in forward function. But since we will pass
            # model.params as arguments, we are ok here.
            return self.model.loss_batch(dnet_bottom_grad, predict)

        gnet_param_arrays = list(self.gnet.params.values())
        gnet_param_keys = list(self.gnet.params.keys())
        gnet_grad_and_loss_func = core.grad_and_loss(
            gnet_loss_func, argnum=range(len(gnet_param_arrays)))
        gnet_grad_arrays, gnet_loss = gnet_grad_and_loss_func(fake_batch.data[0], dnet_bottom_grad, *gnet_param_arrays)
        gnet_grads = dict(zip(gnet_param_keys, gnet_grad_arrays))

        # Perform a parameter update
        for p, w in self.gnet.params.items():
            dw = gnet_grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.gnet.params[p] = next_w
            self.optim_configs[p] = next_config

    def init(self):
        """
        Init model parameters based on the param_configs in model
        """
        for name, config in self.dnet.param_configs.items():
            self.dnet.params[name] = self.init_rules[name](
                config['shape'], self.init_configs[name])
        for name, value in self.dnet.aux_param_configs.items():
            self.dnet.aux_params[name] = value
        for name, config in self.gnet.param_configs.items():
            self.dnet.params[name] = self.init_rules[name](
                config['shape'], self.init_configs[name])
        for name, value in self.gnet.aux_param_configs.items():
            self.gnet.aux_params[name] = value

    def train(self):
        """
        Run optimization to train the model.
        """
        num_iterations = self.train_dataiter.getnumiterations(
        ) * self.num_epochs
        t = 0
        for epoch in range(self.num_epochs):
            start = time.time()
            self.epoch = epoch + 1
            for each_batch in self.train_dataiter:
                rand_batch = self.rand_dataiter.iter_next()
                self._step(each_batch, rand_batch)
                # Maybe print training loss
                if self.verbose and t % self.print_every == 0:
                    print('(Iteration %d / %d) loss: %f' %
                          (t + 1, num_iterations, self.loss_history[-1]))
                t += 1

            # TODO: should call reset automatically
            self._reset_data_iterators()
        # At the end of training swap the best params into the model
        self.model.params = self.best_params
