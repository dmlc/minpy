def forward(self, X):
    a = np.dot(self.params['fc1'], X.T)
    h = np.maximum(0, a)

    # Compute the policy's distribution over actions.
    logits = np.dot(h.T, self.params['policy_fc_last'].T)
    ps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    ps /= np.sum(ps, axis=1, keepdims=True)

    # Compute the value estimates.
    vs = np.dot(h.T, self.params['vf_fc_last'].T) + self.params['vf_fc_last_bias']
    return ps, vs