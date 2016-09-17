class PolicyNetwork(ModelBase):
    def __init__(self,
                 preprocessor,
                 input_size=80*80,
                 hidden_size=200,
                 gamma=0.99):  # Reward discounting factor
        super(PolicyNetwork, self).__init__()
        self.preprocessor = preprocessor
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.add_param('w1', (hidden_size, input_size))
        self.add_param('w2', (1, hidden_size))

    def forward(self, X):
        """Forward pass to obtain the action probabilities for each observation in `X`."""
        a = np.dot(self.params['w1'], X.T)
        h = np.maximum(0, a)
        logits = np.dot(h.T, self.params['w2'].T)
        p = 1.0 / (1.0 + np.exp(-logits))
        return p

    def choose_action(self, p):
        """Return an action `a` and corresponding label `y` using the probability float `p`."""
        a = 2 if numpy.random.uniform() < p else 3
        y = 1 if a == 2 else 0
        return a, y

