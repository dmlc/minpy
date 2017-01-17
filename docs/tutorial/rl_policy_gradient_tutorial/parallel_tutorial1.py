while not all_done:
    # Stack all the observations from the current time step.
    step_xs = np.vstack([o.ravel() for o in observations])

    # Get actions and values for all environments in a single forward pass.
    step_ps, step_vs = agent.forward(step_xs)
    step_as = agent.act(step_ps)

    # Step each environment whose episode has not completed.
    for i, env in enumerate(envs):
        if not done[i]:
            obs, r, done[i], _ = env.step(step_as[i])

            # Record the observation, action, value, and reward in the buffers.
            env_xs[i].append(step_xs[i])
            env_as[i].append(step_as[i])
            env_vs[i].append(step_vs[i])
            env_rs[i].append(r)

            # Store the observation to be used on the next iteration.
            observations[i] = preprocessors[i].preprocess(obs)

    ...