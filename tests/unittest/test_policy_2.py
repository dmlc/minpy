import minpy.numpy as np
import minpy

def test_policy_2():
    np.set_policy(minpy.OnlyMXNetPolicy())
    with minpy.OnlyNumPyPolicy():
        print(np.policy)
        print(np.random.policy)
    print(np.policy)
    print(np.random.policy)

if __name__ == "__main__":
    test_policy_2()
