from __future__ import print_function

import minpy
import minpy.numpy as np
import numpy as npp
from minpy.dispatch.policy import AutoBlacklistPolicy, PreferMXNetPolicy


def test_autoblocklist_policy():
    p = AutoBlacklistPolicy(gen_rule=True, append_rule=True)

    minpy.set_global_policy(p)

    a = np.array([100, 100])
    assert a.dtype == np.int
    assert a == npp.array([100, 100])
    print(a)
    b = np.array([50, 50])
    c = np.array([0, 0])
    np.add(a, b, c)
    assert c.dtype == np.int
    assert c == npp.array([50, 50])
    print(c)
    np.add(a, b, out=c)
    np.add(a, b, out=c)
    assert c.dtype == np.int
    assert c == npp.array([50, 50])
    print(c)
    print('Query primitive [add]')
    output1 = p.query(np, 'add')
    print(output1)
    print('Query primitive [power]')
    output2 = p.query(np, 'power')
    print(output2)
    print('Query primitive [constantinople]')
    output3 = p.query(np, 'constantinople')
    print(output3)

    minpy.set_global_policy(PreferMXNetPolicy())

    result1 = '\n'.join([
        'Total: 2 blacklist rules for primitive [add]:',
        '+-------+------------------------------------+----------------+',
        '|   No. | Type of Positional Args            | Keyword Args   |',
        '+=======+====================================+================+',
        '|     1 | array_dim1, array_dim1, array_dim1 |                |',
        '+-------+------------------------------------+----------------+',
        '|     2 | array_dim1, array_dim1             | out            |',
        '+-------+------------------------------------+----------------+',
    ])
    result2 = 'No rule for power is found in minpy.numpy.'
    result3 = 'minpy.numpy has no attribute constantinople.'
    assert result1 == output1
    assert result2 == output2
    assert result3 == output3


if __name__ == "__main__":
    test_autoblocklist_policy()
