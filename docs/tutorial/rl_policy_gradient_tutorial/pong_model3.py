    def discount_rewards(self, rs):
        drs = np.zeros_like(rs).asnumpy()
        s = 0
        for t in reversed(xrange(0, len(rs))):
            # Reset the running sum at a game boundary.
            if rs[t] != 0:
                s = 0
            s = s * self.gamma + rs[t]
            drs[t] = s
        drs -= np.mean(drs)
        drs /= np.std(drs)
        return drs

class PongPreprocessor(object):
    ... initialization ...
    def preprocess(self, img):
        ... preprocess and flatten the input image...
        return diff
