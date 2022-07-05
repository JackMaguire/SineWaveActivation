import tensorflow as tf

if False:
  from tensorflow.python.keras import backend
  from tensorflow.python.keras import constraints
  from tensorflow.python.keras import initializers
  from tensorflow.python.keras import regularizers
  from tensorflow.python.keras.engine.base_layer import Layer
  from tensorflow.python.keras.engine.input_spec import InputSpec
  from tensorflow.python.keras.utils import tf_utils
else:
  from keras import backend
  from keras import constraints
  from keras import initializers
  from keras import regularizers
  from keras.engine.base_layer import Layer
  from keras.engine.input_spec import InputSpec
  from keras.utils import tf_utils

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.layers.SingleSineWaveActivation')
class SingleSineWaveActivation(Layer):
  def __init__(self,
               alpha_initializer='glorot_uniform',
               alpha_regularizer=None,
               alpha_constraint=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               bias_constraint=None,
               shared_axes=None,
               use_bias=True,
               **kwargs):
    super(SingleSineWaveActivation, self).__init__(**kwargs)
    self.use_bias = use_bias
    self.supports_masking = True
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.alpha_regularizer = regularizers.get(alpha_regularizer)
    self.alpha_constraint = constraints.get(alpha_constraint)
    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)
    if shared_axes is None:
      self.shared_axes = None
    elif not isinstance(shared_axes, (list, tuple)):
      self.shared_axes = [shared_axes]
    else:
      self.shared_axes = list(shared_axes)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    param_shape = list(input_shape[1:])
    if self.shared_axes is not None:
      for i in self.shared_axes:
        param_shape[i - 1] = 1

    self.alpha1 = self.add_weight(
      shape=param_shape,
      name='alpha1',
      initializer=self.alpha_initializer,
      regularizer=self.alpha_regularizer,
      constraint=self.alpha_constraint,
      trainable=True
    )

    self.alpha2 = self.add_weight(
      shape=param_shape,
      name='alpha2',
      initializer=self.alpha_initializer,
      regularizer=self.alpha_regularizer,
      constraint=self.alpha_constraint,
      trainable=True
    )

    self.alpha3 = self.add_weight(
      shape=param_shape,
      name='alpha3',
      initializer=self.alpha_initializer,
      regularizer=self.alpha_regularizer,
      constraint=self.alpha_constraint,
      trainable=True
    )

    if self.use_bias:
      self.bias = self.add_weight(
        name='bias',
        shape=param_shape,
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        dtype=self.dtype)


    # Set input spec
    axes = {}
    if self.shared_axes:
      for i in range(1, len(input_shape)):
        if i not in self.shared_axes:
          axes[i] = input_shape[i]
          self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
          self.built = True

  def call(self, inputs):
    y = tf.math.multiply(
      self.alpha1,
      tf.math.sin( tf.math.add(
        (self.alpha2*inputs*tf.constant(10.0)),
        self.alpha3
      ) )
    )
    if self.use_bias:
      y = tf.math.add( y, self.bias )
    return y

  def get_config(self):
    config = {
      'alpha_initializer': initializers.serialize(self.alpha_initializer),
      'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
      'alpha_constraint': constraints.serialize(self.alpha_constraint),
      'bias_initializer': initializers.serialize(self.bias_initializer),
      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
      'bias_constraint': constraints.serialize(self.bias_constraint),
      'shared_axes': self.shared_axes,
      'use_bias': self.use_bias,
    }
    base_config = super(SingleSineWaveActivation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape

@keras_export('keras.layers.MultipleSineWaveActivation')
class MultipleSineWaveActivation(Layer):
  def __init__(self,
               nwaves:int,
               alpha_initializer='glorot_uniform',
               alpha_regularizer=None,
               alpha_constraint=None,
               bias_initializer='zeros',
               bias_regularizer=None,
               bias_constraint=None,
               shared_axes=None,
               use_bias=True,
               **kwargs):
    super(MultipleSineWaveActivation, self).__init__(**kwargs)

    self.nwaves = nwaves

    self.use_bias = use_bias
    self.supports_masking = True
    self.alpha_initializer = alpha_initializer
    self.alpha_regularizer = alpha_regularizer
    self.alpha_constraint = alpha_constraint
    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)
    if shared_axes is None:
      self.shared_axes = None
    elif not isinstance(shared_axes, (list, tuple)):
      self.shared_axes = [shared_axes]
    else:
      self.shared_axes = list(shared_axes)

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    param_shape = list(input_shape[1:])
    if self.shared_axes is not None:
      for i in self.shared_axes:
        param_shape[i - 1] = 1

    self.waves = []
    for i in range( 0, self.nwaves ):
      self.waves.append( SingleSineWaveActivation(alpha_initializer=self.alpha_initializer, alpha_regularizer=self.alpha_regularizer, alpha_constraint=self.alpha_constraint,use_bias=False) )

    if self.use_bias:
      self.bias = self.add_weight(
        name='bias',
        shape=param_shape,
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        dtype=self.dtype)

    # Set input spec
    axes = {}
    if self.shared_axes:
      for i in range(1, len(input_shape)):
        if i not in self.shared_axes:
          axes[i] = input_shape[i]
          self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
          self.built = True

  def call(self, inputs):
    y = None # self.bias
    for w in self.waves:
      if y == None:
        y = w(inputs)
      else:
        y = tf.math.add( y, w(inputs) )
    if self.use_bias:
      y = tf.math.add( y, self.bias )
    return y

  def get_config(self):
    config = {
      'alpha_initializer': self.alpha_initializer,
      'alpha_regularizer': self.alpha_regularizer,
      'alpha_constraint': self.alpha_constraint,
      'bias_initializer': initializers.serialize(self.bias_initializer),
      'bias_regularizer': regularizers.serialize(self.bias_regularizer),
      'bias_constraint': constraints.serialize(self.bias_constraint),
      'shared_axes': self.shared_axes,
      'use_bias': self.use_bias,
    }
    base_config = super(MultipleSineWaveActivation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape

