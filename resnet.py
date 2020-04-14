from lumen import nil63
from lumen import is63
from lumen import getenv
from absl import flags
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import moving_averages
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-05
FLAGS = flags.FLAGS
def cross_replica_average(inputs=None, num_shards=None, distributed_group_size=None):
  """Calculates the average value of inputs tensor across TPU replicas."""
  __group_assignment = None
  if is63(num_shards) and not( distributed_group_size == num_shards):
    group_size = distributed_group_size
    __group_assignment = []
    for g in range(num_shards // group_size):
      __replica_ids = [g * group_size + i for i in range(group_size)]
      add(__group_assignment, __replica_ids)
  return tpu_ops.cross_replica_sum(inputs, __group_assignment) / tf.cast(distributed_group_size, inputs.dtype)

def distributed_batch_norm(inputs=None, decay=None, epsilon=None, is_training=None, gamma_initializer=None, num_shards=None, distributed_group_size=None, scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

  Note: When is_training is True the moving_mean and moving_variance need to be
  updated, by default the update_ops are placed in `tf.GraphKeys.UPDATE_OPS` so
  they need to be added as a dependency to the `train_op`, example:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops)
      total_loss = control_flow_ops.with_dependencies([updates], total_loss)

  One can set updates_collections=None to force the updates in place, but that
  can have speed penalty, especially in distributed settings.

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension if
    decay: Decay for the moving average. Reasonable values for `decay` are close
      to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
    epsilon: Small float added to variance to avoid dividing by zero.
    is_training: Whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    gamma_initializer:  Initializers for gamma.
    num_shards: Number of shards that participate in the global reduction.
      Default is set to None, that will skip the cross replica sum in and
      normalize across local examples only.
    distributed_group_size: Number of replicas to normalize across in the
      distributed batch normalization.
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.
  """
  if nil63(decay):
    decay = BATCH_NORM_DECAY
  if nil63(epsilon):
    epsilon = BATCH_NORM_EPSILON
  if nil63(is_training):
    is_training = True
  if nil63(gamma_initializer):
    gamma_initializer = None
  if nil63(num_shards):
    num_shards = None
  if nil63(distributed_group_size):
    distributed_group_size = 2
  if nil63(scope):
    scope = None
  __e = None
  with tf1.variable_scope(scope, "batch_normalization", [inputs], reuse=None):
    inputs = tf.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:None]
    __e1 = None
    if not params_shape.is_fully_defined():
      raise ValueError("Inputs %s has undefined `C` dimension %s." % inputs.name % params_shape)
      __e1 = None
    beta = tf1.get_variable("beta", shape=params_shape, dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
    gamma = tf1.get_variable("gamma", shape=params_shape, dtype=tf.float32, initializer=gamma_initializer, trainable=True)
    scope = tf1.get_variable_scope()
    partitioner = scope.partitioner
    scope.set_partitioner(None)
    moving_mean = tf1.get_variable("moving_mean", shape=params_shape, dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = tf1.get_variable("moving_variance", shape=params_shape, initializer=tf.ones_initializer(), trainable=False)
    scope.set_partitioner(partitioner)
    __outputs = None
    __e2 = None
    if is_training:
      axis = 3
      inputs_dtype = inputs.dtype
      inputs = tf.cast(inputs, tf.float32)
      ndims = len(inputs_shape)
      reduction_axes = [i for i in range(ndims) if not( i == axis)]
      counts, mean_ss, variance_ss, _ = tf1.nn.sufficient_statistics(inputs, reduction_axes, keep_dims=False)
      mean_ss = cross_replica_average(mean_ss, num_shards, distributed_group_size)
      variance_ss = cross_replica_average(variance_ss, num_shards, distributed_group_size)
      mean, variance = tf.nn.normalize_moments(counts, mean_ss, variance_ss, shift=None)
      __outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
      __outputs = tf.cast(__outputs, inputs_dtype)
      __e2 = __outputs
    else:
      __outputs, mean, variance = tf.nn.fused_batch_norm(inputs, gamma, beta, mean=moving_mean, variance=moving_variance, epsilon=epsilon, is_training=False, data_format="NHWC")
      __e2 = __outputs, mean, variance
    __e3 = None
    if is_training:
      update_moving_mean = moving_averages.assign_moving_average(moving_mean, tf.cast(mean, moving_mean.dtype), decay, zero_debias=False)
      update_moving_variance = moving_averages.assign_moving_average(moving_variance, tf.cast(variance, moving_variance.dtype), decay, zero_debias=False)
      tf1.add_to_collection("update_ops", update_moving_mean)
      __e3 = tf1.add_to_collection("update_ops", update_moving_variance)
    __outputs.set_shape(inputs_shape)
    __e = __outputs
  return __e

def batch_norm_relu(inputs=None, is_training=None, relu=None, init_zero=None, data_format=None, num_cores=None, distributed_group_size=None):
  """Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either \"channels_first\" for `[batch, channels, height,
        width]` or \"channels_last\" for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`.
  """
  if nil63(relu):
    relu = True
  if nil63(init_zero):
    init_zero = False
  if nil63(data_format):
    data_format = "channels_first"
  if nil63(num_cores):
    num_cores = getenv("num-cores", "value")
  if nil63(distributed_group_size):
    distributed_group_size = getenv("distributed-group-size", "value")
  num_cores = num_cores or 1
  distributed_group_size = distributed_group_size or 2
  __e4 = None
  if init_zero:
    __e4 = tf.zeros_initializer
  else:
    __e4 = tf.ones_initializer
  gamma_initializer = __e4()
  __e5 = None
  if data_format == "channels_first":
    __e5 = 1
  else:
    __e5 = 3
  axis = __e5
  __e6 = None
  if distributed_group_size > 1:
    assert(data_format == "channels_last")
    __e6 = distributed_batch_norm(inputs=inputs, decay=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, is_training=is_training, gamma_initializer=gamma_initializer, num_shards=num_cores, distributed_group_size=distributed_group_size)
  else:
    __e6 = tf1.layers.batch_normalization(inputs=inputs, axis=axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True, scale=True, training=is_training, fused=True, gamma_initializer=gamma_initializer)
  __inputs = __e6
  if relu:
    __inputs = tf.nn.relu(__inputs)
  return __inputs

def fixed_padding(inputs=None, kernel_size=None, data_format=None):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]` or
        `[batch, height, width, channels]` depending on `data_format`.
    kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
        operations. Should be a positive integer.
    data_format: `str` either \"channels_first\" for `[batch, channels, height,
        width]` or \"channels_last\" for `[batch, height, width, channels]`.

  Returns:
    A padded `Tensor` of the same `data_format` with size either intact
    (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
  """
  if nil63(data_format):
    data_format = "channels_first"
  __pad_total = kernel_size - 1
  __pad_beg = __pad_total // 2
  __pad_end = __pad_total - __pad_beg
  if data_format == "channels_first":
    return tf.pad(inputs, [[0, 0], [0, 0], [__pad_beg, __pad_end], [__pad_beg, __pad_end]])
  else:
    return tf.pad(inputs, [[0, 0], [__pad_beg, __pad_end], [__pad_beg, __pad_end], [0, 0]])

def conv2d_fixed_padding(inputs=None, filters=None, kernel_size=None, strides=None, data_format=None):
  """Strided 2-D convolution with explicit padding.

  The padding is consistent and is based only on `kernel_size`, not on the
  dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  Args:
    inputs: `Tensor` of size `[batch, channels, height_in, width_in]`.
    filters: `int` number of filters in the convolution.
    kernel_size: `int` size of the kernel to be used in the convolution.
    strides: `int` strides of the convolution.
    data_format: `str` either \"channels_first\" for `[batch, channels, height,
        width]` or \"channels_last\" for `[batch, height, width, channels]`.

  Returns:
    A `Tensor` of shape `[batch, filters, height_out, width_out]`.
  """
  if nil63(data_format):
    data_format = "channels_first"
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format=data_format)
  __e7 = None
  if strides == 1:
    __e7 = "SAME"
  else:
    __e7 = "VALID"
  return tf1.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=__e7, use_bias=False, kernel_initializer=tf1.variance_scaling_initializer(), data_format=data_format)

def residual_block(inputs=None, filters=None, is_training=None, strides=None, use_projection=None, data_format=None):
  """Standard building block for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either \"channels_first\" for `[batch, channels, height,
        width]` or \"channels_last\" for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block.
  """
  if nil63(use_projection):
    use_projection = False
  if nil63(data_format):
    data_format = "channels_first"
  shortcut = inputs
  if use_projection:
    shortcut = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=strides, data_format=data_format)
    shortcut = batch_norm_relu(shortcut, is_training, relu=False, data_format=data_format)
  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, relu=False, init_zero=True, data_format=data_format)
  return tf.nn.relu(inputs + shortcut)

def bottleneck_block(inputs=None, filters=None, is_training=None, strides=None, use_projection=None, data_format=None):
  """Bottleneck block variant for residual networks with BN after convolutions.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first two convolutions. Note that
        the third and final convolution will use 4 times as many filters.
    is_training: `bool` for whether the model is in training.
    strides: `int` block stride. If greater than 1, this block will ultimately
        downsample the input.
    use_projection: `bool` for whether this block should use a projection
        shortcut (versus the default identity shortcut). This is usually `True`
        for the first block of a block group, which may change the number of
        filters and the resolution.
    data_format: `str` either \"channels_first\" for `[batch, channels, height,
        width]` or \"channels_last\" for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block.
  """
  if nil63(use_projection):
    use_projection = False
  if nil63(data_format):
    data_format = "channels_first"
  shortcut = inputs
  if use_projection:
    filters_out = 4 * filters
    shortcut = conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides, data_format=data_format)
    shortcut = batch_norm_relu(shortcut, is_training, relu=False, data_format=data_format)
  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1, data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
  inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, data_format=data_format)
  inputs = batch_norm_relu(inputs, is_training, relu=False, init_zero=True, data_format=data_format)
  return tf1.nn.relu(inputs + shortcut)

def block_group(inputs=None, filters=None, block_fn=None, blocks=None, strides=None, is_training=None, name=None, data_format=None):
  """Creates one group of blocks for the ResNet model.

  Args:
    inputs: `Tensor` of size `[batch, channels, height, width]`.
    filters: `int` number of filters for the first convolution of the layer.
    block_fn: `function` for the block to use within the model
    blocks: `int` number of blocks contained in the layer.
    strides: `int` stride to use for the first convolution of the layer. If
        greater than 1, this layer will downsample the input.
    is_training: `bool` for whether the model is training.
    name: `str`name for the Tensor output of the block layer.
    data_format: `str` either \"channels_first\" for `[batch, channels, height,
        width]` or \"channels_last\" for `[batch, height, width, channels]`.

  Returns:
    The output `Tensor` of the block layer.
  """
  if nil63(data_format):
    data_format = "channels_first"
  inputs = block_fn(inputs, filters, is_training, strides, use_projection=True, data_format=data_format)
  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, is_training, 1, data_format=data_format)
  return tf.identity(inputs, name)

def resnet_v1(resnet_depth=None, num_classes=None, data_format=None):
  """Returns the ResNet model for a given size and number of output classes."""
  if nil63(data_format):
    data_format = "channels_first"
  model_params = {
    18: {
      "block": residual_block,
      "layers": [2, 2, 2, 2]
    },
    34: {
      "block": residual_block,
      "layers": [3, 4, 6, 3]
    },
    50: {
      "block": bottleneck_block,
      "layers": [3, 4, 6, 3]
    },
    101: {
      "block": bottleneck_block,
      "layers": [3, 4, 23, 3]
    },
    152: {
      "block": bottleneck_block,
      "layers": [3, 8, 36, 3]
    },
    200: {
      "block": bottleneck_block,
      "layers": [3, 24, 36, 3]
    }
  }
  if not( resnet_depth in model_params):
    raise ValueError("Not a valid resnet_depth:", resnet_depth)
  params = model_params[resnet_depth]
  return resnet_v1_generator(params["block"], params["layers"], num_classes, data_format)

def resnet_v1_generator(block_fn=None, layers=None, num_classes=None, data_format=None):
  """Generator for ResNet v1 models.

  Args:
    block_fn: `function` for the block to use within the model. Either
        `residual_block` or `bottleneck_block`.
    layers: list of 4 `int`s denoting the number of blocks to include in each
      of the 4 block groups. Each group consists of blocks that take inputs of
      the same resolution.
    num_classes: `int` number of possible classes for image classification.
    data_format: `str` either \"channels_first\" for `[batch, channels, height,
        width]` or \"channels_last\" for `[batch, height, width, channels]`.

  Returns:
    Model `function` that takes in `inputs` and `is_training` and returns the
    output `Tensor` of the ResNet model.
  """
  if nil63(data_format):
    data_format = "channels_first"
  def model(inputs=None, is_training=None):
    """Creation of the model graph."""
    inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2, data_format=data_format)
    inputs = tf.identity(inputs, "initial-conv")
    inputs = batch_norm_relu(inputs, is_training, data_format=data_format)
    pooled_inputs = tf1.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=2, padding="SAME", data_format=data_format)
    inputs = tf.identity(pooled_inputs, "initial_max_pool")
    inputs = block_group(inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0], strides=1, is_training=is_training, name="block_group1", data_format=data_format)
    inputs = block_group(inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1], strides=2, is_training=is_training, name="block_group2", data_format=data_format)
    inputs = block_group(inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2], strides=2, is_training=is_training, name="block_group3", data_format=data_format)
    inputs = block_group(inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3], strides=2, is_training=is_training, name="block_group4", data_format=data_format)
    pool_size = [inputs.shape[1], inputs.shape[2]]
    inputs = tf1.layers.average_pooling2d(inputs=inputs, pool_size=pool_size, strides=1, padding="VALID", data_format=data_format)
    inputs = tf.identity(inputs, "final_avg_pool")
    __e8 = None
    if block_fn is bottleneck_block:
      __e8 = 2048
    else:
      __e8 = 512
    inputs = tf.reshape(inputs, [-1, __e8])
    inputs = tf1.layers.dense(inputs=inputs, units=num_classes, kernel_initializer=tf1.random_normal_initializer(stddev=0.01))
    return inputs
  model.default_image_size = 224
  return model

def i(x=None):
  return tf.transpose(x, [0, 2, 3, 1])

def o(x=None):
  return tf.transpose(x, [0, 3, 1, 2])

def run_op(op=None, session=None):
  if nil63(session):
    session = getenv("session", "value")
  session.run(tf.global_variables_initializer())
  session.run(tf.local_variables_initializer())
  return session.run(op)

from tensorflow.python.framework.ops import disable_eager_execution
if not( "sess" in globals()):
  global sess
  sess = None
def setup():
  global sess
  disable_eager_execution()
  if sess:
    sess.close()
  sess = tf1.InteractiveSession()
  return sess

