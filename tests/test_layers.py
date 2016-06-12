import numpy as np
import minpy.numpy as mp
import minpy.nn.layers as layers
import minpy.utils.gradient_checker as gradient_checker

def test_affine():
    x = np.random.randn(20, 10)
    b = np.random.randn(20, 1)
    fake_y = np.zeros([20, 5])
    def check_fn(w):
        return layers.l2_loss(layers.affine(x, w, b), fake_y)
    w = np.random.randn(10, 5)
    gradient_checker.quick_grad_check(check_fn, w)

def test_relu():
    fake_y = np.zeros([20, 50])
    def check_fn(x):
        return layers.l2_loss(layers.relu(x), fake_y)
    x = np.random.randn(20, 50)
    gradient_checker.quick_grad_check(check_fn, x)

def test_batchnorm():
    x = np.random.randn(20, 40)
    gamma = np.random.randn(40)
    beta = np.random.randn(40)
    fake_y = np.zeros([20, 40])
    running_mean = np.zeros([40])
    running_var = np.zeros([40])
    def check_mean(mean):
        y = layers.batchnorm(x,
                             gamma,
                             beta,
                             running_mean=mean,
                             running_var=running_var)
        return layers.l2_loss(y, fake_y)
    gradient_checker.quick_grad_check(check_mean, running_mean)
    def check_var(var):
        y = layers.batchnorm(x,
                             gamma,
                             beta,
                             running_mean=running_mean,
                             running_var=var)
        return layers.l2_loss(y, fake_y)
    gradient_checker.quick_grad_check(check_var, running_var)

def main():
    test_affine()
    test_relu()
    test_batchnorm()

if __name__ == '__main__':
    main()
