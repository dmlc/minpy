def loss(self, ps, actions_one_hot, vs, rs, advs):
    # Distribution over actions, prevent log of zero.
    ps = np.maximum(1.0e-5, np.minimum(1.0 - 1e-5, ps))

    # Policy gradient loss.
    policy_grad_loss = -np.sum(np.log(ps) * actions_one_hot * advs)

    # Value function loss.
    vf_loss = 0.5 * np.sum((vs - rs) ** 2)

    # Entropy regularizer.
    entropy = -np.sum(ps * np.log(ps))

    # Weight value function loss and entropy, and combine into the final loss.
    loss_ = policy_grad_loss + self.config.vf_wt * vf_loss - self.config.entropy_wt * entropy
    return loss_