import minpy.numpy as np
import minpy
from minpy.context import cpu, gpu, set_context

def test_policy_2():
    with minpy.OnlyNumPyPolicy():
        print(np.policy)
        print(np.random.policy)
    np.set_policy(minpy.PreferMXNetPolicy())
    set_context(cpu())
    print(np.policy)
    print(np.random.policy)

if __name__ == "__main__":
    test_policy_2()
