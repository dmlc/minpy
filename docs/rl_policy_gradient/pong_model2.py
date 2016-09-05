    def loss(self, ps, ys, rs):
        step_losses = ys * np.log(ps) + (1.0 - ys) * np.log(1.0 - ps)
        return -np.sum(step_losses * rs)
