
(from absl import flags)
(import tensorflow as tf)
(from tensorflow.contrib.tpu.python.ops import tpu-ops)
(from tensorflow.python.training import moving-averages)


(defconst BATCH-NORM-DECAY 0.9)
(defconst BATCH-NORM-EPSILON 1e-5)

(defconst FLAGS (flags .FLAGS))


(def cross-replica-average (inputs num-shards distributed-group-size)
  "Calculates the average value of inputs tensor across TPU replicas."
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
                                    (o is-training True)
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
  (with (tf.variable-scope scope "batch_normalization" (list inputs) reuse: nil)
    (set inputs (tf.convert-to-tensor inputs)
         inputs-shape (inputs (.get-shape))
         params-shape (get inputs-shape (: -1 nil)))
    (unless (params-shape (.is-fully-defined))
      (throw (ValueError (% "Inputs %s has undefined `C` dimension %s." (inputs .name) params-shape))))
    ; Allocate parameters for the beta and gamma of the normalization.
    (set beta (tf.get-variable "beta"
                                shape: params-shape
                                dtype: tf.float32
                                initializer: (tf.zeros-initializer)
                                trainable: true))
    (set gamma (tf.get-variable "gamma"
                                 shape: params-shape
                                 dtype: tf.float32
                                 initializer: gamma-initializer
                                 trainable: true))
    ; Disable partition setting for moving_mean and moving_variance
    ; as assign_moving_average op below doesn't support partitioned variable.
    (set scope (tf.get-variable-scope))
    (set partitioner (scope .partitioner))
    (scope (.set-partitioner nil))
    (set moving-mean (tf.get-variable "moving_mean"
                                       shape: params-shape
                                       dtype: tf.float32
                                       initializer: (tf.zeros-initializer)
                                       trainable: false))
    (set moving-variance (tf.get-variable "moving_variance"
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
               counts, mean-ss, variance-ss, _ (tf.nn.sufficient-statistics inputs reduction-axes keep-dims: false)
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
        (tf.add-to-collection "update-ops" update-moving-mean)
        (tf.add-to-collection "update-ops" update-moving-variance))
      (outputs.set_shape inputs-shape))))

(def batch-norm-relu (inputs is-training (o relu true) (o init-zero false) (o data-format "channels_first")
                             (o num-cores FLAGS.num-cores)
                             (o distributed-group-size FLAGS.distributed-group-size))
  "Performs a batch normalization followed by a ReLU.

  Args:
    inputs: `Tensor` of shape `[batch, channels, ...]`.
    is_training: `bool` for whether the model is training.
    relu: `bool` if False, omits the ReLU operation.
    init_zero: `bool` if True, initializes scale parameter of batch
        normalization with 0 instead of 1 (default).
    data_format: `str` either \"channels_first\" for `[batch, channels, height,
        width]` or \"channels_last\" for `[batch, height, width, channels]`.

  Returns:
    A normalized `Tensor` with the same `data_format`."
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
                   (tf.layers.batch-normalization
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

nil

