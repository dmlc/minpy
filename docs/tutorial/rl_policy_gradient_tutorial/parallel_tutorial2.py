# Perform update and clear buffers. Occurs after `t_max` steps
# or when all episodes are complete.
agent.train_step(env_xs, env_as, env_rs, env_vs)
env_xs, env_as = _2d_list(num_envs), _2d_list(num_envs)
env_rs, env_vs = _2d_list(num_envs), _2d_list(num_envs)