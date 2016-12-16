Show Operation Dispatch Statistics
==================================
In the default ``PreferMXNetPolicy()``, the operation prefers to MXNet implementation and trasparently falls back to NumPy if MXNet hasn't defined it. However, given a network, usually with dozens of operations, it's hard to know which operation is not supported in MXNet and running in NumPy. In some cases, the NumPy running operation is the bottleneck of the program and leads to slow training/inferring speed, especially when MXNet operations are running in GPU. When that happens, a useful speed-up approach is to replace NumPy operation to a MXNet defined one.

To better locate those operations, method ``show_op_stat()`` is provided to show the dispatch statistics information, i.e. which operation and how many operations are executed in MXNet and the same for NumPy. For example, the following network is trained in the default ``PreferMXNetPolicy()``. ``show_op_stat()`` is called after training is finished

::

    from minpy.core import grad
    import minpy.numpy as np
    import minpy.numpy.random as random
    import minpy.dispatch.policy as policy
    
    def test_op_statistics():
    
        def sigmoid(x):
            return 0.5 * (np.tanh(x / 2) + 1)
        
        
        def predict(weights, inputs):
            return sigmoid(np.dot(inputs, weights))
        
        
        def training_loss(weights, inputs):
            preds = predict(weights, inputs)
            label_probabilities = preds * targets + (1 - preds) * (1 - targets)
            l = -np.sum(np.log(label_probabilities))
            return l
        
        
        def training_accuracy(weights, inputs):
            preds = predict(weights, inputs)
            error = np.count_nonzero(
                np.argmax(
                    preds, axis=1) - np.argmax(
                        targets, axis=1))
            return (256 - error) * 100 / 256.0
        
        
        xshape = (256, 500)
        wshape = (500, 250)
        tshape = (256, 250)
        inputs = random.rand(*xshape) - 0.5
        targets = np.zeros(tshape)
        truth = random.randint(0, 250, 256)
        targets[np.arange(256), truth] = 1
        weights = random.rand(*wshape) - 0.5
        
        training_gradient_fun = grad(training_loss)
        
        for i in range(30):
            print('Trained accuracy #{}: {}%'.format(i, training_accuracy(weights,
                                                                          inputs)))
            gr = training_gradient_fun(weights, inputs)
            weights -= gr * 0.01
        
        # Print Op Statistics Info
        np.policy.show_op_stat()
    
    if __name__ == "__main__":
        test_op_statistics()

Here's the statistics result:

::

    MXNET op called times:
      divide : 90
      sum : 30
      negative : 30
      add : 120
      zeros : 1
      multiply : 180
      subtract : 152
      dot : 60
      log : 30

    NUMPY op called times:
      rand : 2
      tanh : 60
      argmax : 60
      randint : 1
      arange : 1
      count_nonzero : 30

    Total Dispatch Proportion: 81.8% in MXNet, 18.2% in NumPy
