Reinforcement learning with policy gradient
===========================================

Deep Reinforcement Learning (RL) is another area where deep models are used. In this example, we
implement an agent that learns to play Pong, trained using policy gradients. Since we are using MinPy,
we avoid the need to manually derive gradient computations, and can easily train on a GPU.

The example is based on the problem and model described in `Deep Reinforcement Learning: Pong from Pixels <http://karpathy.github.io/2016/05/31/rl/>`_ ,
which contains an introduction and background to the ideas discussed here.

The training setups in reinforcement learning often differ based on the approach used, making MinPy's flexibility a great choice for prototyping RL models.
Unlike the standard supervised setting, with the policy gradient approach the training data is generated
during training by the environment and the actions that the agent chooses, and stochasticity is
introduced in the model.

Specifically, we implement a ``PolicyNetwork`` that learns to map states (i.e. visual frames of the Pong game)
to actions (i.e. 'move up' or 'move down'). The network is trained using the ``RLPolicyGradientSolver``.

See `here <https://github.com/dmlc/minpy/blob/master/examples/rl/>`_ for the full implementation.

PolicyNetwork
-------------
The forward pass of the network is separated into two steps:

  1. Compute a probability distribution over actions, given a state.
  2. Choose an action, given the distribution from (1).

These steps are implemented in the ``forward`` and ``choose_action`` functions, respectively:

.. literalinclude:: pong_model1.py
  :language: python
  :linenos:

In the ``forward`` function, we see that the model used is a simple feed-forward network that takes
in an observation ``X`` (a preprocessed image frame) and outputs the probability ``p`` of taking the 'move up' action.

The ``choose_action`` function then draws a random number and selects an action according to the
probability from ``forward``.


For the ``loss`` function, we use cross-entropy but multiply each observation's loss by the associated
discounted reward, which is the key step in forming the policy gradient:

.. literalinclude:: pong_model2.py
  :language: python
  :linenos:

Note that by merely defining the ``forward`` and ``loss`` functions in this way, MinPy will be able to automatically compute the proper gradients.

Lastly, we define the reward discounting approach in ``discount_rewards``, and define the preprocessing
of raw input frames in the ``PongPreprocessor`` class:

.. literalinclude:: pong_model3.py
  :language: python
  :linenos:

See `here <https://github.com/dmlc/minpy/blob/master/examples/rl/policy_gradient/pong_model.py>`_ for the full implementation.

RLPolicyGradientSolver
----------------------
The ``RLPolicyGradientSolver`` is a custom ``Solver`` that can train a model that has the functions discussed above.
In short, its training approach is:

    1. Using the current model, play an episode to generate observations, action labels, and rewards.
    2. Perform a forward and backward pass using the observations, action labels, and rewards from (1).
    3. Update the model using the gradients found in (2).

Under the covers, for step (1) the ``RLPolicyGradientSolver`` repeatedly calls the model's ``forward`` and ``choose_action`` functions at each
step, then uses ``forward`` and ``loss`` functions for step 2.

See `here <https://github.com/dmlc/minpy/blob/master/examples/rl/policy_gradient/rl_policy_gradient_solver.py>`_ for the full implementation.

Training
--------
Run `pong_model.py <https://github.com/dmlc/minpy/blob/master/examples/rl/policy_gradient/pong_model.py>`_ to train the model.
The average running reward is printed, and should generally increase over time. Here is a video
showing the agent (on the right) playing after 18,000 episodes of training:

.. image:: pong.gif
  :align: center
  :width: 50%

While this example uses a very simple feed-forward network, MinPy makes it easy to experiment with more complex architectures,
alternative pre-processing, weight updates, or initializations!

Dependencies
------------------
Note that `Open AI Gym <https://gym.openai.com/>`_ is used for the environment.
