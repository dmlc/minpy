import time
import pickle
import os

import minpy.numpy as np
from minpy import core
from minpy.nn.solver import Solver


class RLPolicyGradientSolver(Solver):
    """A custom `Solver` for models trained using policy gradient.

    Specifically, the model should provide:
        .forward(X)
        .choose_action(p)
        .loss(xs, ys, rs)
        .discount_rewards(rs)
        .preprocessor
    """

    def __init__(self, model, env, **kwargs):
        """ Construct a new `RLPolicyGradientSolver` instance.

        Parameters
        ----------
        model : ModelBase
            A model that supports policy gradient training (see above).
        env : gym.Environment
            A `gym` Environment, e.g. Pong-v0.

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
        self.model = model
        self.env = env
        self.num_episodes = kwargs.pop('num_episodes', 100000)
        self.update_every = kwargs.pop('update_every', 10)
        self.save_every = kwargs.pop('save_every', 10)
        self.save_dir = kwargs.pop('save_dir', './')
        self.resume_from = kwargs.pop('resume_from', None)
        self.render = kwargs.pop('render', False)

        self.running_reward = None
        self.episode_reward = 0

        super(RLPolicyGradientSolver, self).__init__(model, None, None, **kwargs)

    def run_episode(self):
        """Run an episode using the current model to generate training data.

        Specifically, this involves repeatedly getting an observation from the environment,
        performing a forward pass using the single observation to get a distribution over actions
        (in binary case a probability of a single action), and choosing an action.
        Finally, rewards are discounted when the episode completes.

        Returns
        -------
        (xs, ys, rs) : tuple
            The N x input_size observations, N x 1 action labels, and N x 1 discounted rewards
            obtained from running the episode's N steps.
        """
        observation = self.env.reset()
        self.model.preprocessor.reset()
        self.episode_reward = 0

        xs, ys, rs = [], [], []
        done = False
        game_number = 1
        game_start = time.time()
        while not done:
            if self.render:
                self.env.render()
            x = self.model.preprocessor.preprocess(observation)
            p = self.model.forward(x)
            a, y = self.model.choose_action(p.asnumpy().ravel()[0])
            observation, r, done, info = self.env.step(a)

            xs.append(x.asnumpy().ravel())
            ys.append(y)
            rs.append(r)
            self.episode_reward += r
            if self._game_complete(r):
                game_time = time.time() - game_start
                if self.verbose:
                    print('game %d complete (%.2fs), reward: %f' % (game_number, game_time, r))
                game_number += 1
                game_start = time.time()

        # Episode finished.
        self.running_reward = self.episode_reward if self.running_reward is None else (
            0.99*self.running_reward + 0.01*self.episode_reward)
        xs = np.vstack(xs)
        ys = np.vstack(ys)
        rs = np.expand_dims(self.model.discount_rewards(rs), axis=1)
        return xs, ys, rs

    def _game_complete(self, reward):
        return reward != 0

    def train(self):
        """Trains the model for `num_episodes` iterations.

        On each iteration, runs an episode (see `.run_episode()`) to generate three matrices of
        observations, labels and rewards (xs, ys, rs) containing data for the _entire_ episode.
        Then the parameter gradients are found using these episode matrices.

        Specifically, auto-grad is performed on `loss_func`, which does a single forward pass
        with the episode's observations `xs` then computes the loss using the output of the forward
        pass and the episode's labels `ys` and discounted rewards `rs`.

        This two-step approach of generating episode data then doing a single forward/backward pass
        is done to conserve memory during the auto-grad computation.
        """

        # Accumulate gradients since updates are only performed every `update_every` iterations.
        grad_buffer = self._init_grad_buffer()

        for episode_number in xrange(1, self.num_episodes):
            episode_start = time.time()

            # Generate an episode of training data.
            xs, ys, rs = self.run_episode()

            # Performs a forward pass and computes loss using an entire episode's data.
            def loss_func(*params):
                ps = self.model.forward(xs)
                return self.model.loss(ps, ys, rs)

            # Compute gradients with auto-grad on `loss_func` (duplicated from `Solver`).
            param_arrays = list(self.model.params.values())
            param_keys = list(self.model.params.keys())
            grad_and_loss_func = core.grad_and_loss(loss_func, argnum=range(len(param_arrays)))
            backward_start = time.time()
            grad_arrays, loss = grad_and_loss_func(*param_arrays)
            backward_time = time.time() - backward_start
            grads = dict(zip(param_keys, grad_arrays))

            # Accumulate gradients until an update is performed.
            for k, v in grads.iteritems():
                grad_buffer[k] += v

            # Misc. diagnostic info.
            self.loss_history.append(loss.asnumpy())
            episode_time = time.time() - episode_start
            if self.verbose:
                print('Backward pass complete (%.2fs)' % backward_time)
            if self.verbose or episode_number % self.print_every == 0:
                print('Episode %d complete (%.2fs), loss: %s, reward: %s, running reward: %s' %
                      (episode_number, episode_time, loss, self.episode_reward, self.running_reward))

            # Perform parameter update and reset the `grad_buffer` when appropriate.
            if episode_number % self.update_every == 0:
                for p, w in self.model.params.items():
                    dw = grad_buffer[p]
                    config = self.optim_configs[p]
                    next_w, next_config = self.update_rule(w, dw, config)
                    self.model.params[p] = next_w
                    self.optim_configs[p] = next_config
                    grad_buffer[p] = np.zeros_like(w)

            # Save model parameters to `save_dir` when appropriate..
            if episode_number % self.save_every == 0:
                if self.verbose:
                    print('Saving model parameters...')
                file_name = os.path.join(self.save_dir, 'params_%d.p' % episode_number)
                with open(file_name, 'w') as f:
                    pickle.dump({k: v.asnumpy() for k, v in self.model.params.iteritems()}, f)
                if self.verbose:
                    print('Wrote parameter file %s' % file_name)

    def _init_grad_buffer(self):
        return {k: np.zeros_like(v) for k, v in self.model.params.iteritems()}
