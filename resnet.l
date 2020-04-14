(from lumen import nil?)
(from lumen import is?)
(from lumen import getenv)

(from absl import flags)
(import tensorflow as tf)
(import tensorflow.compat.v1 as tf1)
(from tensorflow.python.tpu.ops import tpu-ops)
(from tensorflow.python.training import moving-averages)


(defconst BATCH-NORM-DECAY 0.9)
(defconst BATCH-NORM-EPSILON 1e-5)

(defconst FLAGS (flags .FLAGS))


(def cross-replica-average (inputs num-shards distributed-group-size)
  """Calculates the average value of inputs tensor across TPU replicas."""
  (let group-assignment nil
    (when (and (is? num-shards)
               (not (= distributed-group-size num-shards)))
      (set group-size distributed-group-size
           group-assignment (list))
      (for g in (range (// num-shards group-size))
        (let replica-ids (list (+ (* g group-size) i) for i in (range group-size))
          (add group-assignment replica-ids))))
    (/ (tpu-ops.cross-replica-sum inputs group-assignment)
       (tf.cast distributed-group-size inputs.dtype))))


(def distributed-batch-norm (inputs (o decay BATCH-NORM-DECAY)
                                    (o epsilon BATCH-NORM-EPSILON)
                                    (o is-training true)
                                    (o gamma-initializer nil)
                                    (o num-shards nil)
                                    (o distributed-group-size 2)
                                    (o scope nil))
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
  (with (tf1.variable-scope scope "batch_normalization" (list inputs) reuse: nil)
    (set inputs (tf.convert-to-tensor inputs)
         inputs-shape (inputs (.get-shape))
         params-shape (get inputs-shape (: -1 nil)))
    (unless (params-shape (.is-fully-defined))
      (raise (ValueError (% "Inputs %s has undefined `C` dimension %s." (inputs .name) params-shape))))
    ; Allocate parameters for the beta and gamma of the normalization.
    (set beta (tf1.get-variable "beta"
                                shape: params-shape
                                dtype: tf.float32
                                initializer: (tf.zeros-initializer)
                                trainable: true))
    (set gamma (tf1.get-variable "gamma"
                                 shape: params-shape
                                 dtype: tf.float32
                                 initializer: gamma-initializer
                                 trainable: true))
    ; Disable partition setting for moving_mean and moving_variance
    ; as assign_moving_average op below doesn't support partitioned variable.
    (set scope (tf1.get-variable-scope))
    (set partitioner (scope .partitioner))
    (scope (.set-partitioner nil))
    (set moving-mean (tf1.get-variable "moving_mean"
                                       shape: params-shape
                                       dtype: tf.float32
                                       initializer: (tf.zeros-initializer)
                                       trainable: false))
    (set moving-variance (tf1.get-variable "moving_variance"
                                           shape: params-shape
                                           initializer: (tf.ones-initializer)
                                           trainable: false))
    ; Restore scope's partitioner setting.
    (scope (.set-partitioner partitioner))
    ; Add cross replica sum to do subset mean and variance calculation
    ; First compute mean and variance
    (with outputs nil
      (if is-training
          ; Execute a distributed batch normalization
          (set axis 3
               inputs-dtype (inputs .dtype)
               inputs (tf.cast inputs tf.float32)
               ndims (len inputs-shape)
               reduction-axes (list i for i in (range ndims) if (not (= i axis)))
               counts, mean-ss, variance-ss, _ (tf1.nn.sufficient-statistics inputs reduction-axes keep-dims: false)
               mean-ss (cross-replica-average mean-ss num-shards distributed-group-size)
               variance-ss (cross-replica-average variance-ss num-shards distributed-group-size)
               mean, variance (tf.nn.normalize-moments counts mean-ss variance-ss shift: nil)
               outputs (tf.nn.batch-normalization inputs mean variance beta gamma epsilon)
               outputs (tf.cast outputs inputs-dtype))
          (set outputs, mean, variance (tf.nn.fused-batch-norm
                                         inputs
                                         gamma
                                         beta
                                         mean: moving-mean
                                         variance: moving-variance
                                         epsilon: epsilon
                                         is-training: false
                                         data-format: "NHWC")))
      (when is-training
        (set update-moving-mean
             (moving-averages.assign-moving-average
               moving-mean
               (tf.cast mean moving-mean.dtype)
               decay
               zero-debias: false)
             update-moving-variance
             (moving-averages.assign-moving-average
               moving-variance
               (tf.cast variance moving-variance.dtype)
               decay
               zero-debias: false))
        (tf1.add-to-collection "update_ops" update-moving-mean)
        (tf1.add-to-collection "update_ops" update-moving-variance))
      (outputs.set-shape inputs-shape))))

(def batch-norm-relu (inputs is-training (o relu true) (o init-zero false) (o data-format "channels_first")
                             (t num-cores)
                             (t distributed-group-size))
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
  (set num-cores (or num-cores 1)
       distributed-group-size (or distributed-group-size 2))
  (set gamma-initializer ((if init-zero tf.zeros-initializer tf.ones-initializer)))
  (set axis (if (= data-format "channels_first") 1 3))
  (with inputs (if (> distributed-group-size 1)
                   (do (assert (= data-format "channels_last"))
                       (distributed-batch-norm
                         inputs: inputs
                         decay: BATCH-NORM-DECAY
                         epsilon: BATCH-NORM-EPSILON
                         is-training: is-training
                         gamma-initializer: gamma-initializer
                         num-shards: num-cores
                         distributed-group-size: distributed-group-size))
                   (tf1.layers.batch-normalization
                     inputs: inputs
                     axis: axis
                     momentum: BATCH-NORM-DECAY
                     epsilon: BATCH-NORM-EPSILON
                     center: true
                     scale: true
                     training: is-training
                     fused: true
                     gamma-initializer: gamma-initializer))
    (when relu
      (set inputs (tf.nn.relu inputs)))))


(def fixed-padding (inputs kernel-size (o data-format "channels_first"))
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
  (let (pad-total (- kernel-size 1)
        pad-beg (// pad-total 2)
        pad-end (- pad-total pad-beg))
    (if (= data-format "channels_first")
        (tf.pad inputs `((0 0) (0 0) (,pad-beg ,pad-end) (,pad-beg ,pad-end)))
        (tf.pad inputs `((0 0) (,pad-beg ,pad-end) (,pad-beg ,pad-end) (0 0))))))

(def conv2d-fixed-padding (inputs
                           filters
                           kernel-size
                           strides
                           (o data-format "channels_first"))
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
  (when (> strides 1)
    (set inputs (fixed-padding inputs kernel-size data-format: data-format)))
  (tf1.layers.conv2d
      inputs: inputs
      filters: filters
      kernel-size: kernel-size
      strides: strides
      padding: (if (= strides 1) "SAME" "VALID")
      use-bias: false
      kernel-initializer: (tf1.variance-scaling-initializer)
      data-format: data-format))

(def residual-block (inputs filters is-training strides (o use-projection false) (o data-format "channels_first"))
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
  (set shortcut inputs)
  (when use-projection
    ; Projection shortcut in first layer to match filters and strides
    (set shortcut (conv2d-fixed-padding
                    inputs: inputs
                    filters: filters
                    kernel-size: 1
                    strides: strides
                    data-format: data-format))
    (set shortcut (batch-norm-relu shortcut is-training relu: false data-format: data-format)))

  (set inputs (conv2d-fixed-padding
                inputs: inputs
                filters: filters
                kernel-size: 3
                strides: strides
                data-format: data-format))
  (set inputs (batch-norm-relu inputs is-training data-format: data-format))

  (set inputs (conv2d-fixed-padding
                inputs: inputs
                filters: filters
                kernel-size: 3
                strides: 1
                data-format: data-format))
  (set inputs (batch-norm-relu inputs is-training relu: false init-zero: true data-format: data-format))
  (tf.nn.relu (+ inputs shortcut)))

(def bottleneck_block (inputs filters is_training strides (o use_projection false) (o data_format "channels_first"))
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
  (set shortcut inputs)
  (when use-projection
    ; Projection shortcut only in first block within a group. Bottleneck blocks
    ; end with 4 times the number of filters.
    (set filters-out (* 4 filters))
    (set shortcut (conv2d-fixed-padding
                    inputs: inputs
                    filters: filters_out
                    kernel_size: 1
                    strides: strides
                    data_format: data_format)
         shortcut (batch-norm-relu shortcut is-training relu: false data-format: data-format)))

  (set inputs (conv2d-fixed-padding
                inputs: inputs
                filters: filters
                kernel-size: 1
                strides: 1
                data-format: data-format))
  (set inputs (batch-norm-relu inputs is-training data-format: data-format))

  (set inputs (conv2d-fixed-padding
                inputs: inputs
                filters: filters
                kernel-size: 3
                strides: strides
                data-format: data-format))
  (set inputs (batch-norm-relu inputs is-training data-format: data-format))

  (set inputs (conv2d-fixed-padding
                inputs: inputs
                filters: (* 4 filters)
                kernel-size: 1
                strides: 1
                data-format: data-format))
  (set inputs (batch-norm-relu inputs is-training relu: false init-zero: true data-format: data-format))
  (tf1.nn.relu (+ inputs shortcut)))

(def block-group (inputs filters block_fn blocks strides is_training name (o data_format "channels_first"))
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
  ; Only the first block per block_group uses projection shortcut and strides.
  (set inputs (block-fn inputs filters is-training strides use-projection: true data-format: data-format))
  (for _ in (range 1 blocks)
    (set inputs (block-fn inputs filters is-training 1 data-format: data-format)))
  (tf.identity inputs name))

(def resnet-v1 (resnet-depth num-classes (o data-format "channels_first"))
  """Returns the ResNet model for a given size and number of output classes."""
  (set model-params
       (%object
         18 (obj block: residual-block layers: '(2 2 2 2))
         34 (obj block: residual-block layers: '(3 4 6 3))
         50 (obj block: bottleneck-block layers: '(3 4 6 3))
         101 (obj block: bottleneck-block layers: '(3 4 23 3))
         152 (obj block: bottleneck-block layers: '(3 8 36 3))
         200 (obj block: bottleneck-block layers: '(3 24 36 3))))

  (unless (in resnet-depth model-params)
    (raise (ValueError "Not a valid resnet_depth:" resnet-depth)))

  (set params (get model-params resnet-depth))

  (resnet-v1-generator
    (get params 'block)
    (get params 'layers)
    num-classes
    data-format))

(def resnet-v1-generator (block-fn layers num-classes (o data-format "channels_first"))
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
  (def model (inputs is-training)
    """Creation of the model graph."""

    (set inputs (conv2d-fixed-padding
                  inputs: inputs
                  filters: 64
                  kernel-size: 7
                  strides: 2
                  data-format: data-format)
         inputs (tf.identity inputs "initial-conv")
         inputs (batch-norm-relu inputs is-training data-format: data-format)

         pooled-inputs (tf1.layers.max-pooling2d
                         inputs: inputs pool-size: 3 strides: 2 padding: "SAME"
                         data-format: data-format)
         inputs (tf.identity pooled-inputs "initial_max_pool")

         inputs (block-group
                  inputs: inputs filters: 64 block-fn: block-fn blocks: (at layers 0)
                  strides: 1 is-training: is-training name: "block_group1"
                  data-format: data-format)
         inputs (block-group
                  inputs: inputs filters: 128 block-fn: block-fn blocks: (at layers 1)
                  strides: 2 is-training: is-training name: "block_group2"
                  data-format: data-format)
         inputs (block-group
                  inputs: inputs filters: 256 block-fn: block-fn blocks: (at layers 2)
                  strides: 2 is-training: is-training name: "block_group3"
                  data-format: data-format)
         inputs (block-group
                  inputs: inputs filters: 512 block-fn: block-fn blocks: (at layers 3)
                  strides: 2 is-training: is-training name: "block_group4"
                  data-format: data-format))
    ; The activation is 7x7 so this is a global average pool.
    ; TODO(huangyp): reduce_mean will be faster.
    (set pool-size (list inputs.shape.1 inputs.shape.2))
    (set inputs (tf1.layers.average-pooling2d
                  inputs: inputs pool-size: pool-size strides: 1 padding: "VALID"
                  data-format: data-format)
         inputs (tf.identity inputs "final_avg_pool")
         inputs (tf.reshape inputs (list -1 (if (is block-fn bottleneck-block ) 2048 512)))
         inputs (tf1.layers.dense
                  inputs: inputs
                  units: num-classes
                  kernel-initializer: (tf1.random-normal-initializer stddev: 0.01))))
  (set model.default-image-size 224)
  model)


(def i (x) (tf.transpose x '(0 2 3 1)))
(def o (x) (tf.transpose x '(0 3 1 2)))

(def run-op (op (t session))
  (session.run (tf.global-variables-initializer))
  (session.run (tf.local-variables-initializer))
  (session.run op))

(from tensorflow.python.framework.ops import disable-eager-execution)

(defvar sess nil)

(def setup ()
  (global sess)
  (disable-eager-execution)
  (when sess (sess (.close)))
  (set sess (tf1 (.InteractiveSession))))

