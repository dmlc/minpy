import gym

class Environment:
  def __init__(self, **kwargs):
    self.environment = gym.make(kwargs['environment_name'])    

  def reset(self):
    self.environment.reset()

  def step(self, action):
    return self.environment.step(action)

  @property
  def action_space(self):
    return self.environment.action_space

  @property
  def observation_space(self):
    return self.environment.observation_space

  @property
  def action_range(self):
    return range(self.environment.action_space.n)
