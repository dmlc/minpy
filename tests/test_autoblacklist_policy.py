from __future__ import print_function

import minpy
import minpy.numpy as np

minpy.set_global_policy(minpy.AutoBlacklistPolicy)
minpy.AutoBlacklistPolicy(gen_rule=True, append_rule=True)

a = np.array([100, 100])
print(a)
b = np.array([50, 50])
c = np.array([0, 0])
np.add(a, b, c)
print(c)
np.add(a, b, out=c)
np.add(a, b, out=c)
print(c)
