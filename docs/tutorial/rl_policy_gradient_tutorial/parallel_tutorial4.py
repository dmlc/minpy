for i in range(num_envs):
    # Compute discounted rewards with a 'bootstrapped' final value.
    rs_bootstrap = [] if env_rs[i] == [] else env_rs[i] + [env_vs[i][-1]]
    discounted_rewards.extend(self.discount(rs_bootstrap, gamma)[:-1])

    # Compute advantages for each environment using Generalized Advantage Estimation;
    # see eqn. (16) of [Schulman 2016].
    delta_t = env_rs[i] + gamma*env_vs[i][1:] - env_vs[i][:-1]
    advantages.extend(self.discount(delta_t, gamma*lambda_))