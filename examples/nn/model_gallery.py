import minpy.nn.model_builder as builder
'''
  Network In Network

  Reference:
  Min Lin, Qiang Chen, Shuicheng Yan, Network In Network
'''
network_in_network = builder.Sequential(
  builder.Reshape((3, 32, 32)),
  builder.Convolution((5, 5), 192, pad=(2, 2)),
  builder.ReLU(),
  builder.Convolution((1, 1), 160),
  builder.ReLU(),
  builder.Convolution((1, 1), 96),
  builder.ReLU(),
  builder.Pooling('max', (3, 3), (2, 2)),
  builder.Dropout(0.5),
  builder.Convolution((5, 5), 192, pad=(2, 2)),
  builder.ReLU(),
  builder.Convolution((1, 1), 192),
  builder.ReLU(),
  builder.Convolution((1, 1), 192),
  builder.ReLU(),
  builder.Pooling('avg', (3, 3), (2, 2)),
  builder.Dropout(0.5),
  builder.Convolution((3, 3), 192, pad=(1, 1)),
  builder.ReLU(),
  builder.Convolution((1, 1), 192),
  builder.ReLU(),
  builder.Convolution((1, 1), 10),
  builder.ReLU(),
  builder.Pooling('avg', (8, 8)),
  builder.Reshape((10,))
)

'''
  Residual Network

  Reference:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
'''
def residual_network(n):
  '''
    n: the network contains 6 * n + 2 layers, including 6 * n + 1 convolution layers and 1 affine layer
       please refer to the paper for details
  '''
  def normalized_convolution(kernel_shape, kernel_number, stride, pad, activate=None):
    module = builder.Sequential(
      builder.Convolution(kernel_shape, kernel_number, stride, pad),
      builder.SpatialBatchNormalization()
    )
    if activate:
      module.append(getattr(builder, activate)())
    return module

  def residual(kernel_number, project=False):
    if project:
      module = builder.Add(
        builder.Sequential(
          normalized_convolution((3, 3), kernel_number, (2, 2), (1, 1), 'ReLU'),
          normalized_convolution((3, 3), kernel_number, (1, 1), (1, 1))
        ),
        builder.Sequential(
          builder.Pooling('avg', (2, 2), (2, 2)),
          builder.Convolution((1, 1), kernel_number)
        )
      )
    else:
      module = builder.Add(
        builder.Sequential(
          normalized_convolution((3, 3), kernel_number, (1, 1), (1, 1), 'ReLU'),
          normalized_convolution((3, 3), kernel_number, (1, 1), (1, 1))
        ),
        builder.Identity()
      )
    return module

  network = builder.Sequential(
    builder.Reshape((3, 32, 32)),
    normalized_convolution((3, 3), 16, (1, 1), (1, 1), 'ReLU')
  )
  for i in range(n):
    network.append(residual(16))
  network.append(residual(32, project=True))
  for i in range(n-1):
    network.append(residual(32))
  network.append(residual(64, project=True))
  for i in range(n-1):
    network.append(residual(64))
  network.append(builder.Pooling('avg', (8, 8)))
  network.append(builder.Reshape((64,)))
  network.append(builder.Affine(10))

  return network
