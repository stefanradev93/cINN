import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2


class CouplingNet(tf.keras.Model):
    """Implements a conditional version of a sequential network."""

    def __init__(self, meta, n_out):
        """
        Creates a conditional coupling net (FC neural network).
        Arguments:
        meta  : list -- a list of dictionaries, wherein each dictionary holds parameter - value pairs for a single
                       tf.keras.Dense layer.
        n_out : int  -- number of outputs of the coupling net
        """

        super(CouplingNet, self).__init__()

        self.dense = tf.keras.Sequential(
            # Hidden layer structure
            [tf.keras.layers.Dense(units,
                                   activation=meta['activation'],
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))
             for units in meta['n_units']] +
            # Output layer
            [tf.keras.layers.Dense(n_out,
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))]
        )

    def call(self, x, y):
        """
        Concatenates x and y and performs a forward pass through the coupling net.
        Arguments:
        x : tf.Tensor of shape (batch_size, inp_dim)     -- the parameters x ~ p(x|y) of interest
        y : tf.Tensor of shape (batch_size, summary_dim) -- the summarized conditional data of interest y = sum(y)
        """

        inp = tf.concat((x, y), axis=-1)
        out = self.dense(inp)
        return out


class ConditionalInvertibleBlock(tf.keras.Model):
    """Implements a conditional version of the INN block."""

    def __init__(self, meta, x_dim, alpha=1.9, permute=False):
        """
        Creates a conditional invertible block.
        ----------

        Arguments:
        meta  : list -- a list of dictionaries, wherein each dictionary holds parameter - value pairs for a single
                       tf.keras.Dense layer. All coupling nets are assumed to be equal.
        x_dim : int  -- the number of outputs of the invertible block (eq. the dimensionality of the latent space)
        alpha : float or None -- used to do soft clamping ot the outputs (loss smoothing)
        """
        super(ConditionalInvertibleBlock, self).__init__()

        self.alpha = alpha
        self.n_out1 = x_dim // 2
        self.n_out2 = x_dim // 2 if x_dim % 2 == 0 else x_dim // 2 + 1
        if permute:
            self.permutation_vec = np.random.permutation(x_dim)
            self.inv_permutation_vec = np.argsort(self.permutation_vec)
        else:
            self.permutation_vec = None
            self.inv_permutation_vec = None
        self.s1 = CouplingNet(meta, self.n_out1)
        self.t1 = CouplingNet(meta, self.n_out1)
        self.s2 = CouplingNet(meta, self.n_out2)
        self.t2 = CouplingNet(meta, self.n_out2)

    def call(self, x, y, inverse=False, log_det_J=True):
        """
        Implements both directions of a conditional invertible block.
        ----------

        Arguments:
        x         : tf.Tensor of shape (batch_size, inp_dim) -- the parameters x ~ p(x|y) of interest
        y         : tf.Tensor of shape (batch_size, summary_dim) -- the summarized conditional data of interest y = sum(y)
        inverse   : bool -- flag indicating whether to tun the block forward or backwards
        log_det_J : bool -- flag indicating whether to return the log determinant of the Jacobian matrix
        ----------

        Output:
        (v, log_det_J)  :  (tf.Tensor of shape (batch_size, inp_dim), tf.Tensor of shape (batch_size, )) --
                           the transformed input, if inverse = False, and the corresponding Jacobian of the transformation
                            if inverse = False
        u               :  tf.Tensor of shape (batch_size, inp_dim) -- the transformed out, if inverse = True
        """

        # --- Forward pass --- #
        if not inverse:

            if self.permutation_vec is not None:
                x = tf.transpose(tf.gather(tf.transpose(x), self.permutation_vec))

            u1, u2 = tf.split(x, [self.n_out1, self.n_out2], axis=-1)

            # Pre-compute network outputs for v1
            s1 = self.s1(u2, y)
            # Clamp s1 if specified
            if self.alpha is not None:
                s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
            t1 = self.t1(u2, y)
            v1 = u1 * tf.exp(s1) + t1

            # Pre-compute network outputs for v2
            s2 = self.s2(v1, y)
            # Clamp s2 if specified
            if self.alpha is not None:
                s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
            t2 = self.t2(v1, y)
            v2 = u2 * tf.exp(s2) + t2
            v = tf.concat((v1, v2), axis=-1)

            if log_det_J:
                # log|J| = log(prod(diag(J))) -> according to inv architecture
                return v, tf.reduce_sum(s1, axis=-1) + tf.reduce_sum(s2, axis=-1)
            return v

        # --- Inverse pass --- #
        else:

            v1, v2 = tf.split(x, [self.n_out1, self.n_out2], axis=-1)

            # Pre-Compute s2
            s2 = self.s2(v1, y)
            # Clamp s2 if specified
            if self.alpha is not None:
                s2 = (2. * self.alpha / np.pi) * tf.math.atan(s2 / self.alpha)
            u2 = (v2 - self.t2(v1, y)) * tf.exp(-s2)

            # Pre-Compute s1
            s1 = self.s1(u2, y)
            # Clamp s1 if specified
            if self.alpha is not None:
                s1 = (2. * self.alpha / np.pi) * tf.math.atan(s1 / self.alpha)
            u1 = (v1 - self.t1(u2, y)) * tf.exp(-s1)
            u = tf.concat((u1, u2), axis=-1)

            if self.permutation_vec is not None:
                u = tf.transpose(tf.gather(tf.transpose(u), self.inv_permutation_vec))

            return u


class DeepConditionalModel(tf.keras.Model):
    """Implements a chain of conditional invertible blocks."""

    def __init__(self, meta, n_blocks, x_dim, summary_net=None, alpha=1.9, permute=False):
        """
        Creates a chain of cINN blocks and chains operations.
        ----------

        Arguments:
        meta        : list -- a list of dictionary, where each dictionary holds parameter - value pairs for a single
                                  keras.Dense layer
        n_blocks    : int  -- the number of invertible blocks
        x_dim       : int  -- the dimensionality of the space to be learned
        summary_net : tf.keras.Model or None -- an optinal summary network for learning the sumstats of y
        alpha       : float or None -- used to do soft clamping ot the outputs (loss smoothing)
        permute     : bool -- whether to permute the inputs to the cINN
        """

        super(DeepConditionalModel, self).__init__()
        
        self.cINNs = [ConditionalInvertibleBlock(meta, x_dim, alpha, permute) for _ in range(n_blocks)]
        self.summary_net = summary_net
        self.x_dim = x_dim

    def call(self, x, y, inverse=False):
        """
        Performs one pass through an invertible chain (either inverse or forward).
        ----------

        Arguments:
        x         : tf.Tensor of shape (batch_size, inp_dim) -- the parameters x ~ p(x|y) of interest
        y         : tf.Tensor of shape (batch_size, summary_dim) -- the summarized conditional data of interest y = summary(y)
        inverse   : bool -- flag indicating whether to tun the chain forward or backwards
        ----------

        Output:
        (z, log_det_J)  :  (tf.Tensor of shape (batch_size, inp_dim), tf.Tensor of shape (batch_size, )) --
                           the transformed input, if inverse = False, and the corresponding Jacobian of the transformation
                            if inverse = False
        x               :  tf.Tensor of shape (batch_size, inp_dim) -- the transformed out, if inverse = True
        """

        if self.summary_net is not None:
            y = self.summary_net(y)
        if inverse:
            return self.inverse(x, y)
        else:
            return self.forward(x, y)

    def forward(self, x, y):
        """Performs a forward pass though the chain."""

        z = x
        log_det_Js = []
        for cINN in self.cINNs:
            z, log_det_J = cINN(z, y)
            log_det_Js.append(log_det_J)
        # Sum Jacobian determinants for all blocks to obtain total Jacobian.
        log_det_J = tf.add_n(log_det_Js)
        return z, log_det_J

    def inverse(self, x, y):
        for cINN in reversed(self.cINNs):
            x = cINN(x, y, inverse=True)
        return x

    def sample(self, y, n_samples, to_numpy=False):
        """
        Samples from the inverse model given a single instance y or a batch of instances.
        ----------
        
        Arguments:
        y         : tf.Tensor of shape (batch_size, summary_dim) -- the summarized conditional data of interest y = summary(y)
        n_samples : int -- number of samples to obtain from the approximate posterior
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        ----------

        Returns:
        X_samples : 3D tf.Tensor or np.array of shape (n_samples, n_batch, x_dim)
        """

        # Summarize obs data if summary net available
        if self.summary_net is not None:
            y = self.summary_net(y)

        # In case y is a single instance
        if y.shape[0] == 1:
            z_normal_samples = tf.random_normal(shape=(n_samples, self.x_dim), dtype=tf.float32)

            # Sample batch by repeating y over the batch dimension
            X_samples = self.inverse(z_normal_samples, tf.tile(y, [n_samples, 1]))

        # In case of a batch input, send a 3D tensor through the invertible chain and use tensordot
        # Warning: This tensor could get pretty big if sampling a lot of values for a lot of batch instances!
        else:
            z_normal_samples = tf.random_normal(shape=(n_samples, y.shape[0], self.x_dim), dtype=tf.float32)

            # Sample batch by repeating y over the batch dimension
            X_samples = self.inverse(z_normal_samples, tf.stack([y] * n_samples))
        if to_numpy:
            return X_samples.numpy()
        return X_samples