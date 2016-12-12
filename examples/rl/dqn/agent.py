import minpy.numpy as np
import minpy.nn.model_builder as builder
import minpy.nn.init as initializer
import minpy.nn.optim as optimizer

from environment import *

class Agent:
  def __init__(self, environment, **kwargs):
    self.environment = environment

    self.batch_size    = kwargs['batch_size']
    self.episodes      = kwargs['episodes']
    self.maximum_steps = kwargs['maximum_steps']
    self.sample_number = kwargs['sample_number']
    self.sync_interval = kwargs['sync_interval']

    observation_shape = self.environment.observation_space.shape

    # observation_shape = self.environment.reset().shape
    self.history_capacity = kwargs['history_capacity']
    self.history = [
      {
        'action'      : None,
        'observation' : np.zeros(observation_shape),
        'reward'      : None
      } \
        for i in range(self.history_capacity)
    ]
    self.history_to_update = 0
    self.__observation_cache = np.zeros(observation_shape)

    # construct neural networks
    network_structure = builder.Sequential(
      builder.Convolution((8, 8), 32, (4, 4)),
      builder.ReLU(),
      builder.Convolution((4, 4), 64, (2, 2)),
      builder.ReLU(),
      builder.Convolution((3, 3), 64),
      builder.ReLU(),
      builder.Flatten(),
      builder.Affine(512),
      builder.ReLU(),
      builder.Affine(len(self.environment.action_range))
    )

    self.q_network = builder.Model(network_structure, None, (4, 84, 84))
    self.target_network = builder.Model(network_structure, None, (4, 84, 84))
    for key, value in self.q_network.param_configs.items():
      self.q_network.params[key] = \
        getattr(initializer, value['init_rule'])(
          value['shape']
          value.pop('config', {})
        )
      self.target_network.params[key] = self.q_network.params[key][:]    


  def update_history(self, action, observation, reward):
    self.__observation_cache[:] = self.history[self.history_to_update]['observation']

    self.history[self.history_to_update]['observation'][:] = observation
    self.history[self.history_to_update]['action']         = action
    self.history[self.history_to_update]['reward']         = reward

    self.history_to_update = (self.history_to_update + 1) % self.history_capacity

  def extract_history(self, index):
    if index == self.history_to_update:
      previous_observation = self.__observation_cache
    else:
      index = (index + self.history_capacity - 1) % self.history_capacity
      previous_observation = self.history[index]['observation']
    return \
      previous_observation, \
      self.history[index]['observation'], \
      self.history[index]['action'], \
      self.history[index]['reward']

  def random_action(self):
    return self.environment.action_space.sample()

  def predict(self, observation):
    return self.random_action()

  def train(self):
    observation = self.environment.reset()
    for episode in range(self.episodes):
      for step in range(self.maximum_steps):
        action = self.predict(observation)
        observation, reward, finished, _ = self.environment.step(action)
        self.update_history(observation, reward, finished)

        # choose a subset of history data
        index = np.random.choice(range(len(self.history)), self.sample_number, False)

        if finished:
          break
