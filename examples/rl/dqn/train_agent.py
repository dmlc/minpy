from agent import *
from environment import *

environment_settings = {
  'environment_name' : 'Breakout-v0'
}

agent_settings = {
  'batch_size'       : 32,
  'episodes'         : 32,
  'history_capacity' : 32,
  'maximum_steps'    : 32,
  'sample_number'    : 32,
}

agent = Agent(
  Environment(**environment_settings),
  **agent_settings
)

agent.train()
