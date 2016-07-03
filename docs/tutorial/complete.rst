Complete solver and optimizer guide
===================================

The layer->model->solver (optimizer) pipeline

Flow: 
- Description of the general architecture
- Describe the role of the architectural components: layer (operator), model, solver and optimizer
- A diagram of the architecture
- Description of the folders/paths (i.e. where to find the components)

Example 1: two layer fully connected, with batch normalization and drop out (i.e. cs231n); fairly complete code snippets, with link to the full code

Example 2: replace some of the bottom layers with convolution, using mxnet symbolic computation; link to the code. Keypoint: only show the changes in the model and layer part; keypoint is solver/optimizer need not be touched

Example 3: image caption, with RNN and convnet (taking the convnet from example 2), show the change in model and layer; link to the complete code

Wrap up. Touch also on advanced topics:
- How to deal with multiple losses, possibly attached to different model segments?
- What if different model segments are to use different learning rates?
- ....
