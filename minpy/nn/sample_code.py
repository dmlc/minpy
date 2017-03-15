class ModelUpdater:
  ...

# 3 approaches to initialize updater
# specify optimizer setting for every parameter
optimizer_settings = {
  'fc0_weight' : {'optimizer' : 'Adam', 'lr' : 0.1, 'beta' : 0.9, 'epsilon' : 0.9},
  'fc0_bias'   : {'optimizer' : 'SGD', 'lr' : 0.1},
  'fc1_weight' : {'optimizer' : 'Adam', 'lr' : 0.1, 'beta' : 0.9, 'epsilon' : 0.9},
  'fc1_bias'   : {'optimizer' : 'SGD', 'lr' : 0.1},
}
# 'broadcast' one optimizer setting to all parameters in one group of parameters
optimizer_settings = {
  ('fc0_weight', 'fc0_bias') : {'optimizer' : 'adam', 'lr' : 0.1, 'beta' : 0.9, 'epsilon' : 0.9},
  ('fc1_weight', 'fc1_bias') : {'optimizer' : 'SGD', 'lr' : 0.1},
}
# 'broadcast' one optimizer setting to all parameters
optimizer_settings = {'optimizer' : 'Adam', 'lr' : 0.1, 'beta' : 0.9, 'epsilon' : 0.9}

model_updater = ModelUpdater(model, optimizer_settings)

# one step of update
model_updater.update(gradient_map)                             # update all parameters
model_updater['fc0_weight'].update(gradient_map['fc0_weight']) # update fc0_weight

# user could change hyperparameters of optimizer at any time
model_updater.lr *= 0.1               # decay learning rate of all parameters
model_updater['fc0_weight'].lr *= 0.1 # decay learning rate of fc0_weight

'''
Cache is internal state of optimizer.
It is unnecessary for users to manage it mannually, but users could access it.
'''
print model_updater.cache
print model_updater['fc0_weight'].cache
