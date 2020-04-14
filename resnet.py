from lumen import nil63
from lumen import is63
from lumen import getenv
from lumen import setenv
from lumen import either
from lumen import unstash
from lumen import destash33
from lumen import has
from lumen import object
from lumen import without
from lumen import cut
from absl import flags
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import moving_averages
def expanded_body(body=None):
  __form1 = expand(join(["%do"], body))
  if hd63(__form1, "%do"):
    return tl(__form1)
  else:
    return [__form1]

def compiled_special63(form=None):
  return list63(form) and (hd63(form, string63) and (char(hd(form), 0) == "%" and (L_35(hd(form)) > 1 and not( char(hd(form), 1) == "%"))))

def __f3(name=None, *_args, **_keys):
  ____r5 = unstash(list(_args), _keys)
  __name1 = destash33(name, ____r5)
  ____id1 = ____r5
  __e5 = None
  if nil63(has(____id1, "scope")):
    __e5 = "nil"
  else:
    __e5 = has(____id1, "scope")
  __scope1 = __e5
  __e6 = None
  if nil63(has(____id1, "reuse")):
    __e6 = "nil"
  else:
    __e6 = has(____id1, "reuse")
  __reuse1 = __e6
  ____x11 = object([])
  ____x11["scope"] = ["o", "scope", ["quote", "nil"]]
  ____x11["reuse"] = ["o", "reuse", ["quote", "nil"]]
  ____x11["rest"] = "body"
  __body1 = without(cut(____id1, 0), ____x11)
  ____x17 = object(["tf1.variable-scope", __scope1, __name1])
  ____x17["reuse"] = __reuse1
  return join(["with", ____x17], __body1)

setenv("tf-scope", macro=__f3)
def __f4(inputs=None, name=None, *_args, **_keys):
  ____r8 = unstash(list(_args), _keys)
  __inputs1 = destash33(inputs, ____r8)
  __name3 = destash33(name, ____r8)
  ____id3 = ____r8
  __e7 = None
  if nil63(has(____id3, "scope")):
    __e7 = "nil"
  else:
    __e7 = has(____id3, "scope")
  __scope3 = __e7
  __e8 = None
  if nil63(has(____id3, "reuse")):
    __e8 = "nil"
  else:
    __e8 = has(____id3, "reuse")
  __reuse3 = __e8
  ____x30 = object([])
  ____x30["scope"] = ["o", "scope", ["quote", "nil"]]
  ____x30["reuse"] = ["o", "reuse", ["quote", "nil"]]
  ____x30["rest"] = "body"
  __body3 = without(cut(____id3, 0), ____x30)
  ____x36 = object(["tf1.variable-scope", __scope3, __name3])
  ____x36["reuse"] = __reuse3
  def __f5(x=None):
    if compiled_special63(x):
      return x
    else:
      return ["set", __inputs1, ["either", x, __inputs1]]
  return join(["with", ____x36], map(__f5, expanded_body(__body3)), [["set", __inputs1, ["tf.identity", __inputs1, __name3]]])

setenv("tf-named", macro=__f4)
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
    distributed_group_size = 1
  if nil63(scope):
    scope = None
  inputs = tf.convert_to_tensor(inputs)
  inputs_shape = inputs.get_shape()
  params_shape = inputs_shape[-1:None]
  if not params_shape.is_fully_defined():
    raise ValueError("Inputs %s has undefined `C` dimension %s." % inputs.name % params_shape)
  __e9 = None
  with tf1.variable_scope(scope, "batch_normalization", reuse=None):
    beta = tf1.get_variable("beta", shape=params_shape, dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
    gamma = tf1.get_variable("gamma", shape=params_shape, dtype=tf.float32, initializer=gamma_initializer, trainable=True)
    scope = tf1.get_variable_scope()
    partitioner = scope.partitioner
    scope.set_partitioner(None)
    moving_mean = tf1.get_variable("moving_mean", shape=params_shape, dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
    moving_variance = tf1.get_variable("moving_variance", shape=params_shape, initializer=tf.ones_initializer(), trainable=False)
    scope.set_partitioner(partitioner)
    __outputs = None
    __e10 = None
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
      __e10 = __outputs
    else:
      __outputs, mean, variance = tf.nn.fused_batch_norm(inputs, gamma, beta, mean=moving_mean, variance=moving_variance, epsilon=epsilon, is_training=False, data_format="NHWC")
      __e10 = __outputs, mean, variance
    __e11 = None
    if is_training:
      update_moving_mean = moving_averages.assign_moving_average(moving_mean, tf.cast(mean, moving_mean.dtype), decay, zero_debias=False)
      update_moving_variance = moving_averages.assign_moving_average(moving_variance, tf.cast(variance, moving_variance.dtype), decay, zero_debias=False)
      tf1.add_to_collection("update_ops", update_moving_mean)
      __e11 = tf1.add_to_collection("update_ops", update_moving_variance)
    __outputs.set_shape(inputs_shape)
    __e9 = __outputs
  return __e9

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
  distributed_group_size = distributed_group_size or 1
  __e12 = None
  if init_zero:
    __e12 = tf.zeros_initializer
  else:
    __e12 = tf.ones_initializer
  gamma_initializer = __e12()
  __e13 = None
  if data_format == "channels_first":
    __e13 = 1
  else:
    __e13 = 3
  axis = __e13
  __e14 = None
  if distributed_group_size > 1:
    assert(data_format == "channels_last")
    __e14 = distributed_batch_norm(inputs=inputs, decay=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, is_training=is_training, gamma_initializer=gamma_initializer, num_shards=num_cores, distributed_group_size=distributed_group_size)
  else:
    __e14 = tf1.layers.batch_normalization(inputs=inputs, axis=axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True, scale=True, training=is_training, fused=True, gamma_initializer=gamma_initializer)
  __inputs2 = __e14
  if relu:
    __inputs2 = tf.nn.relu(__inputs2)
  return __inputs2

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
  __e15 = None
  if strides == 1:
    __e15 = "SAME"
  else:
    __e15 = "VALID"
  return tf1.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=__e15, use_bias=False, kernel_initializer=tf1.variance_scaling_initializer(), data_format=data_format)

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

def block_group(inputs=None, filters=None, block_fn=None, blocks=None, strides=None, is_training=None, name=None, data_format=None, scope=None):
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
  if nil63(scope):
    scope = None
  with tf1.variable_scope(scope, name, [inputs], reuse=None):
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
    with tf1.variable_scope(None, "initial-conv", reuse=None):
      inputs = either(conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2, data_format=data_format), inputs)
      inputs = either(batch_norm_relu(inputs, is_training, data_format=data_format), inputs)
      inputs = tf.identity(inputs, "initial-conv")
    with tf1.variable_scope(None, "initial_max_pool", reuse=None):
      inputs = either(tf1.layers.max_pooling2d(inputs=inputs, pool_size=3, strides=2, padding="SAME", data_format=data_format), inputs)
      inputs = tf.identity(inputs, "initial_max_pool")
    inputs = block_group(inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0], strides=1, is_training=is_training, name="block_group1", data_format=data_format)
    inputs = block_group(inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1], strides=2, is_training=is_training, name="block_group2", data_format=data_format)
    inputs = block_group(inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2], strides=2, is_training=is_training, name="block_group3", data_format=data_format)
    inputs = block_group(inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3], strides=2, is_training=is_training, name="block_group4", data_format=data_format)
    pool_size = [inputs.shape[1], inputs.shape[2]]
    with tf1.variable_scope(None, "final_avg_pool", reuse=None):
      inputs = either(tf1.layers.average_pooling2d(inputs=inputs, pool_size=pool_size, strides=1, padding="VALID", data_format=data_format), inputs)
      inputs = tf.identity(inputs, "final_avg_pool")
    __e16 = None
    with tf1.variable_scope(None, "final_dense", reuse=None):
      __e4 = None
      __e17 = None
      if block_fn is bottleneck_block:
        __e4 = 2048
        __e17 = __e4
      else:
        __e4 = 512
        __e17 = __e4
      inputs = either(tf.reshape(inputs, [-1, __e4]), inputs)
      inputs = either(tf1.layers.dense(inputs=inputs, units=num_classes, kernel_initializer=tf1.random_normal_initializer(stddev=0.01)), inputs)
      inputs = tf.identity(inputs, "final_dense")
      __e16 = inputs
    return __e16
  model.default_image_size = 224
  return model

def i(x=None):
  return tf.transpose(x, [0, 2, 3, 1])

def o(x=None):
  return tf.transpose(x, [0, 3, 1, 2])

def run_op(op=None, *_args, **_keys):
  ____r23 = unstash(list(_args), _keys)
  __op = destash33(op, ____r23)
  ____id4 = ____r23
  __e18 = None
  if nil63(has(____id4, "session")):
    __e18 = tf1.get_default_session()
  else:
    __e18 = has(____id4, "session")
  __session = __e18
  ____x65 = object([])
  ____x65["session"] = ["o", "session", ["tf1.get-default-session"]]
  ____x65["rest"] = "keys"
  __keys = without(cut(____id4, 0), ____x65)
  return __session.run(__op, **__keys)

from tensorflow.python.framework.ops import disable_eager_execution
if not( "sess" in globals()):
  global sess
  sess = None
def setup(graph=None):
  if nil63(graph):
    graph = tf1.Graph()
  global sess
  disable_eager_execution()
  if sess:
    sess.close()
  sess = tf1.InteractiveSession(graph=graph)
  return sess

def test_resnet(num_classes=None, shape=None, model_size=None, data_format=None):
  if nil63(num_classes):
    num_classes = 10
  if nil63(shape):
    shape = [1, 28, 28, 3]
  if nil63(model_size):
    model_size = 50
  if nil63(data_format):
    data_format = "channels_last"
  setup()
  __ph = tf1.placeholder(tf.float32, shape=shape)
  __net = resnet_v1(model_size, num_classes, data_format=data_format)(__ph, is_training=True)
  run_op(tf1.global_variables_initializer())
  run_op(tf1.local_variables_initializer())
  return [__ph, __net]

