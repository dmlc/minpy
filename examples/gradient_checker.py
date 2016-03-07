from minpy.core import grad
import minpy.numpy as np
import minpy.numpy.random as random
import minpy.array
import itertools as it
from minpy.dispatch import registry
import numpy as nnp

EPS, RTOL, ATOL = 1e-1, 1e-1, 1e-1

def quick_grad_check(fun, arg0, extra_args=(), kwargs={}, verbose=True,
                     eps=EPS, rtol=RTOL, atol=ATOL, rs=None):
    """Checks the gradient of a function (w.r.t. to its first arg) in a random direction"""

    if verbose:
        print("Checking gradient of {0} at {1}".format(fun, arg0))

    if rs is None:
        rs = nnp.random.RandomState()
    
    random_dir = rs.standard_normal(nnp.shape(arg0))
    random_dir = random_dir / nnp.sqrt(nnp.sum(random_dir * random_dir))
 
    if not extra_args == ():
      unary_fun = lambda x : fun(arg0 + x * random_dir, extra_args)
      numeric_grad = (unary_fun(eps/2) - unary_fun(-eps/2)) / eps
      analytic_grad = np.sum(grad(fun)(arg0, extra_args) * random_dir)
    else:
      unary_fun = lambda x : fun(arg0 + x * random_dir)
      numeric_grad = (unary_fun(eps/2) - unary_fun(-eps/2)) / eps
      analytic_grad = np.sum(grad(fun)(arg0) * random_dir)
  
    if isinstance(numeric_grad, minpy.array.Number):
        assert abs((analytic_grad - numeric_grad).get_data(None)) < atol and abs((analytic_grad - numeric_grad).get_data(None)) < abs((analytic_grad * rtol).get_data(None)), \
            "Check failed! nd={0}, ad={1}".format(numeric_grad, analytic_grad)
    elif isinstance(numeric_grad, minpy.array.Array):
        assert abs((analytic_grad - numeric_grad).asnumpy()) < atol and abs((analytic_grad - numeric_grad).asnumpy()) < abs((analytic_grad * rtol).asnumpy()), \
            "Check failed! nd={0}, ad={1}".format(numeric_grad, analytic_grad)
    else:
        assert False
    if verbose:
        print("Gradient projection OK (numeric grad: {0}, analytic grad: {1})".format(
            numeric_grad, analytic_grad))

def Check_All_Func():
  reg = minpy.array.Value._ns._registry._reg
  for each_key in reg.keys():
    for each_type in reg[each_key]:
      if (reg[each_key][each_type].grad_defined()):
        print each_type, reg[each_key][each_type]
      







