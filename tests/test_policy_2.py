import minpy.numpy as np
import minpy

np.set_policy(minpy.OnlyMXNetPolicy)
with minpy.OnlyNumPyPolicy:
    print(np.policy)
    print(np.random.policy)
print(np.policy)
print(np.random.policy)
