import time
import pickle
import os

import minpy.numpy as np
from minpy import core
from minpy.nn.solver import Solver
from minpy.nn.io import DataBatch
from minpy.nn import optim, init

class DCGanSolver(Solver):
    """A custom `Solver` for Discriminative Generative GAN models

    Specifically, the model should provide:
        .forward(X)
        .loss(xs, ys, rs)
    """

    def __init__(self, gnet, dnet, train_iter, rand_iter, **kwargs):
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
        self.real_dataiter = train_iter
        self.rand_dataiter = rand_iter
        self.num_episodes = kwargs.pop('num_episodes', 100000)
        self.save_every = kwargs.pop('save_every', 10)
        self.save_dir = kwargs.pop('save_dir', './')
        self.resume_from = kwargs.pop('resume_from', None)
        self.render = kwargs.pop('render', False)
        self.verbose = True

        super(DCGanSolver, self).__init__(gnet, None, None, **kwargs)

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.dnet_loss_history = []
        self.gnet_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.real_dataiter.reset()
        self.rand_dataiter.reset()
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

        def dnet_loss_func(batch_data, batch_label, *params): # pylint: disable=unused-argument
            """
            Loss function calculate the loss
            """

            # It seems that params are not used in forward function. But since we will pass
            # model.params as arguments, we are ok here.
            predict = self.dnet.forward_batch(batch_data, mode='train')
            return self.dnet.loss_batch(batch_label, predict)

        dnet_param_arrays = list(self.dnet.params.values())
        dnet_param_keys = list(self.dnet.params.keys())
        dnet_grad_and_loss_func = core.grad_and_loss(
            dnet_loss_func, argnum=range(len(dnet_param_arrays) + 2))
        
        dnet_grad_arrays_real, dnet_loss_real = dnet_grad_and_loss_func(real_batch.data[0], real_batch.label[0],  *dnet_param_arrays)
        self.dnet_loss_history.append(dnet_loss_real[0])
        print 'dnet real'
        print real_batch.label[0]
        print dnet_loss_real[0]
        
        dnet_grads_real = dict(zip(dnet_param_keys, dnet_grad_arrays_real[2:]))

        ##########################
        # (2) train DNet with fake
        ##########################
        #generated_data = self.gnet.forward_batch(rand_batch[0], mode='train').as_numpy()
        generated_data = self.gnet.forward_batch(rand_batch[0], mode='train')
        fake_batch = DataBatch([generated_data], [np.zeros(generated_data.shape[0])])
        
        dnet_grad_arrays_fake, dnet_loss_fake = dnet_grad_and_loss_func(fake_batch.data[0], fake_batch.label[0], *dnet_param_arrays)
        self.dnet_loss_history[-1] += dnet_loss_fake[0]
        print 'dnet_fake'
        print fake_batch.label[0]
        print dnet_loss_fake[0]
        dnet_grads_fake = dict(zip(dnet_param_keys, dnet_grad_arrays_fake[2:]))


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
        fake_batch = DataBatch([generated_data], [np.ones(generated_data.shape[0])])
        dnet_grad_and_loss_func_on_data = core.grad_and_loss(
            dnet_loss_func, argnum=range(1))
        
        dnet_grad_arrays_on_data, dnet_loss_on_data = dnet_grad_and_loss_func_on_data(fake_batch.data[0], fake_batch.label[0], *dnet_param_arrays)
        self.gnet_loss_history.append(dnet_loss_on_data[0])
        dnet_bottom_grad = dnet_grad_arrays_on_data[0]

        def gnet_loss_func(rand_batch_data, dnet_bottom_grad, *params): # pylint: disable=unused-argument
            """
            Loss function calculate the loss
            """

            # It seems that params are not used in forward function. But since we will pass
            # model.params as arguments, we are ok here.
            gnet_out = self.gnet.forward_batch(rand_batch_data, mode='train')
            return self.model.loss_batch(dnet_bottom_grad, gnet_out)

        gnet_param_arrays = list(self.gnet.params.values())
        gnet_param_keys = list(self.gnet.params.keys())
        gnet_grad_and_loss_func = core.grad_and_loss(
            gnet_loss_func, argnum=range(len(gnet_param_arrays)+2))
        gnet_grad_arrays, gnet_loss = gnet_grad_and_loss_func(rand_batch[0], dnet_bottom_grad, *gnet_param_arrays)
        gnet_grads = dict(zip(gnet_param_keys, gnet_grad_arrays[2:]))

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
            self.gnet.params[name] = self.init_rules[name](
                config['shape'], self.init_configs[name])
        for name, value in self.gnet.aux_param_configs.items():
            self.gnet.aux_params[name] = value

    def train(self):
        """
        Run optimization to train the model.
        """
        num_iterations = self.real_dataiter.getnumiterations() * self.num_epochs
        t = 0
        for epoch in range(self.num_epochs):
            start = time.time()
            self.epoch = epoch + 1
            for each_batch in self.real_dataiter:
                rand_batch = self.rand_dataiter.getdata()
                self._step(each_batch, rand_batch)
                # Maybe print training loss
                print('(Iteration %d / %d, dloss %f, gloss %f)' %
                      (t + 1, num_iterations, self.dnet_loss_history[-1], self.gnet_loss_history[-1]))
                t += 1

            # TODO: should call reset automatically
            self._reset_data_iterators()
