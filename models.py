import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2


class Permutation(tf.keras.Model):
    """Implements a permutation layer to permute the input dimensions of the cINN block."""

    def __init__(self, theta_dim):
        """
        Creates a permutation layer for a conditional invertible block.
        ----------

        Arguments:
        theta_dim  : int -- the dimensionality of the input to the c inv block.
        """

        super(Permutation, self).__init__()

        permutation_vec = np.random.permutation(theta_dim)
        inv_permutation_vec = np.argsort(permutation_vec)
        self.permutation = tf.Variable(initial_value=permutation_vec,
                                       trainable=False, 
                                       dtype=tf.int32,
                                       name='permutation')
        self.inv_permutation = tf.Variable(initial_value=inv_permutation_vec,
                                           trainable=False, 
                                           dtype=tf.int32,
                                           name='inv_permutation')

    def call(self, x, inverse=False):
        """Permutes the bach of an input."""

        if not inverse:
            return tf.transpose(tf.gather(tf.transpose(x), self.permutation))
        return tf.transpose(tf.gather(tf.transpose(x), self.inv_permutation))


class CouplingNet(tf.keras.Model):
    """Implements a conditional version of a sequential network."""

    def __init__(self, meta, n_out):
        """
        Creates a conditional coupling net (FC neural network).
        ----------

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
            self.permutation = Permutation(x_dim)
        else:
            self.permutation = None
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

            if self.permutation is not None:
                x = self.permutation(x)

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

            if self.permutation is not None:
                u = self.permutation(u, inverse=True)
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
        # Sum Jacobian determinants for all blocks to obtain total log det(Jacobian).
        log_det_J = tf.add_n(log_det_Js)
        return z, log_det_J

    def inverse(self, x, y):
        for cINN in reversed(self.cINNs):
            x = cINN(x, y, inverse=True)
        return x

    def sample(self, y, n_samples, to_numpy=False, training=False):
        """
        Samples from the inverse model given a single instance y or a batch of instances.
        ----------
        
        Arguments:
        y         : tf.Tensor of shape (batch_size, summary_dim) -- the summarized conditional data of interest y = summary(y)
        n_samples : int -- number of samples to obtain from the approximate posterior
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        training  : bool -- flag used to indicate that samples are drawn are training time (BatchNorm)
        ----------

        Returns:
        X_samples : 3D tf.Tensor or np.array of shape (n_samples, n_batch, x_dim)
        """

        # Summarize obs data if summary net available
        if self.summary_net is not None:
            y = self.summary_net(y, training=training)

        # In case y is a single instance
        if int(y.shape[0]) == 1:
            z_normal_samples = tf.random_normal(shape=(n_samples, self.x_dim), dtype=tf.float32)

            # Sample batch by repeating y over the batch dimension
            X_samples = self.inverse(z_normal_samples, tf.tile(y, [n_samples, 1]))

        # In case of a batch input, send a 3D tensor through the invertible chain and use tensordot
        # Warning: This tensor could get pretty big if sampling a lot of values for a lot of batch instances!
        else:
            z_normal_samples = tf.random_normal(shape=(n_samples, int(y.shape[0]), self.x_dim), dtype=tf.float32)

            # Sample batch by repeating y over the batch dimension
            X_samples = self.inverse(z_normal_samples, tf.stack([y] * n_samples))
        if to_numpy:
            return X_samples.numpy()
        return X_samples


class InvariantModule(tf.keras.Model):
    """Implements an invariant nn module as proposed by Bloem-Reddy and Teh (2019)."""

    def __init__(self, h_dim, n_dense=3):
        """
        Creates an invariant function with mean pooling.
        ----------

        Arguments:
        h_dim   : int -- the number of hidden units in each of the modules
        n_dense : int -- the number of dense layers of the modules
        """
        
        super(InvariantModule, self).__init__()
        
        self.module = tf.keras.Sequential([
            tf.keras.layers.Dense(h_dim, activation='elu', kernel_initializer='glorot_uniform') 
            for _ in range(n_dense)
        ])

        self.post_pooling_dense = tf.keras.layers.Dense(h_dim, activation='elu', kernel_initializer='glorot_uniform')   
        
    def call(self, x):
        """
        Transofrms the input into an invariant representation.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or 'samples' dimensions 
            over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, h_dim) -- the pooled and invariant representation of the input
        """

        x = self.module(x)
        x = tf.reduce_mean(x, axis=1)
        out = self.post_pooling_dense(x)
        return out


class InvariantModuleLearnable(tf.keras.Model):
    """Implements an invariant nn module as proposed by Bloem-Reddy and Teh (2019)."""

    def __init__(self, h_dim, n_dense=3):
        """
        Creates an invariant function with mean pooling.
        ----------

        Arguments:
        meta : dict -- a dictionary with hyperparameter name - values
        """

        super(InvariantModuleLearnable, self).__init__()


        self.module = tf.keras.Sequential([
            tf.keras.layers.Dense(h_dim, activation='elu', kernel_initializer='glorot_uniform') 
            for _ in range(n_dense)
        ])
        self.weights_layer = tf.keras.Sequential([
                tf.keras.layers.Dense(h_dim, activation='elu', kernel_initializer='glorot_uniform')
                for _ in range(n_dense)
            ] + 
            [
                tf.keras.layers.Dense(h_dim, kernel_initializer='glorot_uniform')
            
            ])
        self.post_pooling_dense = tf.keras.Sequential([
            tf.keras.layers.Dense(h_dim, activation='elu', kernel_initializer='glorot_uniform')
            for _ in range(h_dim)
        ])

    def call(self, x):
        """
        Transofrms the input into an invariant representation.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or 'samples' dimensions
            over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, h_dim) -- the pooled and invariant representation of the input
        """

        # Embed
        x_emb = self.module(x)

        # Compute weights
        w = tf.nn.softmax(self.weights_layer(x), axis=1)
        w_x = tf.reduce_sum(x_emb * w, axis=1)
    
        # Increase representational power
        out = self.post_pooling_dense(w_x)
        return out
    

class EquivariantModule(tf.keras.Model):
    """Implements an equivariant nn module as proposed by Bloem-Reddy and Teh (2019)."""

    def __init__(self, h_dim, n_dense=3, learnable_pooling=False, inv_xdim=False):
        """
        Creates an equivariant neural network consisting of a FC network with
        equal number of hidden units in each layer and an invariant module
        with the same FC structure.
        ----------

        Arguments:
        h_dim   : int -- the number of hidden units in each of the modules
        n_dense : int -- the number of dense layers of the modules
        """
        
        super(EquivariantModule, self).__init__()
        
        self.module = tf.keras.Sequential([
            tf.keras.layers.Dense(h_dim, activation='elu') 
            for _ in range(n_dense)
        ])
        
        if learnable_pooling:
            self.invariant_module = InvariantModuleLearnable(h_dim, n_dense)
        else:
            self.invariant_module = InvariantModule(h_dim, n_dense)
        
        
    def call(self, x):
        """
        Transofrms the input into an equivariant representation.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or 'samples' dimensions 
            over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, h_dim) -- the pooled and invariant representation of the input
        """

        x_inv = self.invariant_module(x)
        x_inv = tf.stack([x_inv] * int(x.shape[1]), axis=1) # Repeat x_inv n times
        x = tf.concat((x_inv, x), axis=-1)
        out = self.module(x)
        return out
    

class InvariantNetwork(tf.keras.Model):
    """
    Implements a network which parameterizes a 
    permutationally invariant function according to Bloem-Reddy and Teh (2019).
    """

    def __init__(self, h_dim, n_dense=3, n_equiv=2, learnable_pooling=False):
        """
        Creates a permutationally invariant network 
        consisting of two equivariant modules and one invariant module.
        ----------

        Arguments:
        h_dim   : int -- the number of hidden units in each of the modules
        n_dense : int -- the number of dense layers of the modules
        n_equiv : int -- the number of equivariant modules 
        """
        
        super(InvariantNetwork, self).__init__()
        
        self.equiv = tf.keras.Sequential([
            EquivariantModule(h_dim, n_dense, learnable_pooling=learnable_pooling)
            for _ in range(n_equiv)
        ])

        if learnable_pooling:
            self.inv = InvariantModuleLearnable(h_dim, n_dense)
        else:
            self.inv = InvariantModule(h_dim, n_dense)
        
    def call(self, x, **kwargs):
        """
        Transofrms the input into a permutationally invariant 
        representation by first passing it through multiple equivariant 
        modules in order to increase representational power.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or 
        'samples' dimensions over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, h_dim) -- the pooled and invariant representation of the input
        """

        x = self.equiv(x)
        out = self.inv(x)
        return out


class SelfAttentionHead(tf.keras.Model):
    """Implements a single self-attention head."""

    def __init__(self, key_dim=64, dense_dim=128):
        super(SelfAttentionHead, self).__init__()
        
        self.Wq = tf.keras.layers.Dense(key_dim, use_bias=False, kernel_initializer='glorot_uniform')
        self.Wk = tf.keras.layers.Dense(key_dim, use_bias=False, kernel_initializer='glorot_uniform')
        self.Wv = tf.keras.layers.Dense(key_dim, use_bias=False, kernel_initializer='glorot_uniform')
        
        self.dense_part = tf.keras.models.Sequential([
            tf.keras.layers.Dense(dense_dim, activation='elu', kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(dense_dim, activation='elu', kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(dense_dim, activation='elu', kernel_initializer='glorot_uniform'),
        ])
        
        
        self.scale = np.sqrt(key_dim)
              
    def call(self, x):
        """
        Passes the input through an attention head and then through a shared dense layer.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or 'samples' dimensions 
            over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : tf.Tensor of shape (batch_size, n, h_dim) -- the trasnformed representation of the input
        """

        # Obtain queries, keys, and values
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)
        
        # Compute self-attention matrices
        qk_t = tf.einsum('bij,bkj->bik', queries, keys)
        weights = tf.nn.softmax(qk_t / self.scale, axis=-1)
        z = tf.einsum('bik,bkj->bij', weights, values)
        
        # Pass through final dense
        out = self.dense_part(z)
        return out


class AttentionNetwork(tf.keras.Model):
    """
    Implements a network with multi-head self-attention architecture after Vaswani et al. (2017).
    """

    def __init__(self, n_heads=8, key_dim=64, dense_dim=128):
        """
        Creates an attention network composed of multiplke heads and a pooling layer
        consisting of two equivariant modules and one invariant module.
        ----------

        Arguments:
        n_heads   : int -- the number of attention heads
        key_dim   : int -- the dimensionality of the attention keys
        dense_dim : int -- the number of dense layers for the final part of the network
        """
        super(AttentionNetwork, self).__init__()
        
        self.Wo = tf.keras.layers.Dense(key_dim, use_bias=False, kernel_initializer='glorot_uniform')
        self.attention_heads = [SelfAttentionHead(key_dim) for _ in range(n_heads)]
        
        self.dense_part = tf.keras.models.Sequential([
            tf.keras.layers.Dense(dense_dim, activation='elu', kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(dense_dim, activation='elu', kernel_initializer='glorot_uniform'),
            tf.keras.layers.Dense(dense_dim, activation='elu', kernel_initializer='glorot_uniform'),
        ])
        
    def call(self, x, **args):
        """
        Passes the input through multiple attention head, pools the output of the multiple heads 
        and then passed the pooled output through a dense layer.
        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, n, m) - the input where n is the 'time' or 'samples' dimensions 
            over which pooling is performed and m is the input dimensionality
        ----------

        Returns:
        out : the trasnformed input after attention, pooling and non-linearity
        """

        # Calculate attention
        x = tf.concat([head(x) for head in self.attention_heads], axis=-1)
        x = self.Wo(x)
        
        # Pooling
        x = tf.reduce_sum(x, axis=1)
        
        # Non-lineariy after pooling
        out = self.dense_part(x)
        return out


class ConditionalVAE(tf.keras.Model):
    """Implements a conditional variational autoencoder."""
    def __init__(self, meta, theta_dim, z_dim, summary_net=None):
        """
        Creates a conditional variation autoencoder cVAE
        ----------

        Arguments:
        meta        : list -- a list of dictionary, where each dictionary holds parameter - value pairs for a single
                                  keras.Dense layer
        theta_dim   : int  -- the dimensionality of the parameter space to be learned
        z_dim       : int  -- the dimensionality of the latent space
        summary_net : tf.keras.Model or None -- an optinal summary network for learning the sumstats of y
        """
        super(ConditionalVAE, self).__init__()
        
        self.summary_net = summary_net
        self.z_dim = z_dim
        
        # Encoder networks
        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units,
                                   activation=meta['activation'],
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))
             for units in meta['n_units']]
        )
        
        self.z_mapper = tf.keras.layers.Dense(
           z_dim * 2, 
           kernel_initializer=meta['initializer'],
           kernel_regularizer=l2(meta['w_decay'])
        )
        
        # Decoder networks 
        self.decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units,
                                   activation=meta['activation'],
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))
             for units in meta['n_units']]
        )

        # Network to output the reconstructed parameters
        self.inference_net = tf.keras.layers.Dense(
            theta_dim, 
            kernel_initializer=meta['initializer'],
            kernel_regularizer=l2(meta['w_decay'])
        )
        
    def call(self, x, y):
        """
        Computes a summary of h(y), concatenates x and h(y) and passes them through encoder and decoder.
        ----------
        
        Arguments:
        x : tf.Tensor of shape (batch_size, inp_dim)     -- the parameters x ~ p(x|y) of interest
        y : tf.Tensor of shape (batch_size, summary_dim) -- the conditional data of interest y = sum(y)
        
        ----------
        
        Output:
        z_mean   : tf.Tensor of shape (batch_size, z_dim) -- the means of the latent Gaussian distribution
        z_logvar : tf.Tensor of shape (batch_size, z_dim) -- the logvars of the latent Gaussian distribution
        x_rec    : tf.Tensor of shape (batch_size, theta_dim) -- the reconstructed parameters
        """
        
        # Compute summary, if summary net has been given
        if self.summary_net is not None:
            y = self.summary_net(y)
        
        # Encode
        z_mean, z_logvar = self.encode(x, y)
        
        # Sample latent
        eps = tf.random_normal(shape=z_mean.shape)
        z = z_mean + eps * tf.exp(z_logvar * 0.5)
        
        # Decode x
        x_rec = self.decode(z, y)
        
        return z_mean, z_logvar, x_rec
    
    def encode(self, x, y):
        """Encodes x given y into a latent mean and var."""
        
        x = tf.concat([x, y], axis=-1)
        x = self.encoder(x)
        x = self.z_mapper(x)
        z_mean, z_logvar = tf.split(x, 2, axis=-1)
        return z_mean, z_logvar
    
    def decode(self, z, y):
        """Decodes x from z given y."""
        
        z_c = tf.concat([z, y], axis=-1)
        x_rec = self.decoder(z_c)
        x_rec = self.inference_net(x_rec)
        return x_rec
    
    def sample(self, y, n_samples, to_numpy=False, training=False):
        """
        Samples from the decoder given a single instance y or a batch of instances.
        ----------
        
        Arguments:
        y         : tf.Tensor of shape (batch_size, n_points) -- the conditional data of interest
        n_samples : int -- number of samples to obtain from the approximate posterior
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        ----------

        Returns:
        theta_samples : 3D tf.Tensor or np.array of shape (n_samples, n_batch, theta_dim)
        """
        
        # Compute summary, if summary network given
        if self.summary_net is not None:
            y = self.summary_net(y, training=training)
        
        # Sample from unit Gaussian
        z = tf.random_normal(shape=(y.shape[0], n_samples, self.z_dim))
        
        # Concatenate samples and conditionals and decode
        z_c = tf.concat([z,  tf.stack([y] * n_samples, axis=1)], axis=-1)
        theta_samples = self.inference_net(self.decoder(z_c))
        theta_samples = tf.transpose(theta_samples, [1, 0, 2])
        if to_numpy:
            return theta_samples.numpy()
        return theta_samples


class IAFConditionalVAE(tf.keras.Model):
    """
    Implements a conditional IAF-VAE according to Kingma et al. (2016).
    """

    def __init__(self, meta, theta_dim, z_dim, n_iaf, summary_net=None, reverse_z=False):
        """
        Creates a conditional variation autoencoder cVAE with an inverse autoregressive flow (IAF)
        ----------

        Arguments:
        meta        : list -- a list of dictionary, where each dictionary holds parameter - value pairs for a single
                                  keras.Dense layer
        theta_dim   : int  -- the dimensionality of the parameter space to be learned
        z_dim       : int  -- the dimensionality of the latent space
        n_iaf       : int  -- the number of autoregressive neural networks
        summary_net : tf.keras.Model or None -- an optinal summary network for learning the sumstats of y
        """
        super(IAFConditionalVAE, self).__init__()

        self.summary_net = summary_net
        self.z_dim = z_dim
        self.reverse_z = reverse_z

        # Encoder networks
        self.encoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units,
                                   activation=meta['activation'],
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))
             for units in meta['n_units']]
        )
        self.z_h_mapper = tf.keras.layers.Dense(z_dim * 3,
                                   activation=meta['activation'],
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))

        # Autoregressive networks
        self.ar_nns = [tf.keras.Sequential(
            [
                tf.keras.layers.Dense(meta['n_units'][0],
                                     kernel_initializer='ones',
                                     activation=meta['activation'],
                                     kernel_regularizer=l2(meta['w_decay'])

            )] +
            [
                tf.keras.layers.Dense(z_dim * 2,
                                  kernel_initializer='ones',
                                  kernel_regularizer=l2(meta['w_decay'])
                )
            ])
        for _ in range(n_iaf)
        ]

        # Decoder networks
        self.decoder = tf.keras.Sequential(
            [tf.keras.layers.Dense(units,
                                   activation=meta['activation'],
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))
             for units in meta['n_units']]
        )

        # Network to output the reconstructed parameters
        self.inference_net = tf.keras.layers.Dense(
            theta_dim,
            kernel_initializer=meta['initializer'],
            kernel_regularizer=l2(meta['w_decay'])
        )

    def call(self, x, y):
        """
        Computes a summary of h(y), concatenates x and h(y) and passes
        them through encoder ar nets and decoder. Computes log(q(z|x)) along the way.

        ----------

        Arguments:
        x : tf.Tensor of shape (batch_size, inp_dim)     -- the parameters x ~ p(x|y) of interest
        y : tf.Tensor of shape (batch_size, summary_dim) -- the conditional data of interest y = sum(y)

        ----------

        Output:
        l        : tf.Tensor of shape (batch_size, ) -- the value of log(q(z|x))
        x_rec    : tf.Tensor of shape (batch_size, theta_dim) -- the reconstructed parameters
        """


        # Compute summary, if summary net has been given
        if self.summary_net is not None:
            y = self.summary_net(y)

        # Encode x, and condition
        z0_mean, z0_logsigma, h  = self.encode(x, y)

        # Sample epsilon and reparameterize to z
        eps = tf.random_normal(shape=(z0_mean.shape[0], z0_mean.shape[1]))
        z = z0_mean + eps * tf.exp(z0_logsigma)

        # Compute first step of l
        l = -tf.reduce_sum(z0_logsigma + 0.5 * tf.square(eps) + 0.5 * tf.log(2 * np.pi), axis=-1)

        # Autoregressive step
        for t, ar_nn in enumerate(self.ar_nns):

            # Reverse z every other step
            if t % 2 != 0 and self.reverse_z:
                z = tf.reverse(z, [-1])

            # Concat z, y, and h as input to arnn and obtain real valued vectors m, s
            z_h = tf.concat([z, y, h], axis=-1)
            m, s = tf.split(ar_nn(z_h), 2, axis=-1)

            # "Gating" step
            sigma = tf.nn.sigmoid(s)
            z = sigma * z + (1 - sigma) * m

            # Increment value of log(q(z|x))
            l -= tf.reduce_sum(tf.log(sigma + 1e-8), axis=-1)

        # Decode final z^T
        x_rec = self.decode(z, y)

        return x_rec, z, l

    def encode(self, x, y):
        """Encodes x given y into a latent mean and var."""

        x = tf.concat([x, y], axis=-1)
        x = self.encoder(x)
        x = self.z_h_mapper(x)
        z0_mean, z0_logsigma, h = tf.split(x, 3, axis=-1)
        return z0_mean, z0_logsigma, h

    def decode(self, z, y):
        """Decodes x from z given y."""

        z_c = tf.concat([z, y], axis=-1)
        x_rec = self.decoder(z_c)
        x_rec = self.inference_net(x_rec)
        return x_rec

    def sample(self, y, n_samples, to_numpy=False, training=False):
        """
        Samples from the decoder given a single instance y or a batch of instances.
        ----------

        Arguments:
        y         : tf.Tensor of shape (batch_size, n_points) -- the conditional data of interest
        n_samples : int -- number of samples to obtain from the approximate posterior
        to_numpy  : bool -- flag indicating whether to return the samples as a np.array or a tf.Tensor
        ----------

        Returns:
        theta_samples : 3D tf.Tensor or np.array of shape (n_samples, n_batch, theta_dim)
        """

        # Compute summary, if summary network given
        if self.summary_net is not None:
            y = self.summary_net(y, training=training)

        # Sample from unit Gaussian
        z = tf.random_normal(shape=(y.shape[0], n_samples, self.z_dim))

        # Concatenate z and conditionals and decode
        theta_samples = self.decode(z, tf.stack([y] * n_samples, axis=1))
        theta_samples = tf.transpose(theta_samples, [1, 0, 2])

        if to_numpy:
            return theta_samples.numpy()
        return theta_samples


class HeteroscedasticModel(tf.keras.Model):
    """
    Creates a heteroscedastric regression model according to Kendall & Gal (2017).
    """

    def __init__(self, meta, theta_dim, summary_net=None):
        """
        Creates a heteroscedastic model for parameter inference.
        Arguments:
        meta        : list -- a list of dictionary, where each dictionary holds parameter - value pairs for a single
                                  keras.Dense layer
        theta_dim   : int  -- the dimensionality of the parameter space on which to perform inference
        summary_net : tf.keras.Model or None -- an optinal summary network for learning the sumstats of y
        """
        super(HeteroscedasticModel, self).__init__()
        self.summary_net = summary_net
        self.inference_net = tf.keras.Sequential(
            [tf.keras.layers.Dense(units,
                                   activation=meta['activation'],
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))
             for units in meta['n_units']] +

            [tf.keras.layers.Dense(theta_dim * 2,
                                   kernel_initializer=meta['initializer'],
                                   kernel_regularizer=l2(meta['w_decay']))]
        )

    def call(self, y):
        """
        Computes a summary of h(y), concatenates x and h(y) and passes them through encoder and decoder.
        ----------

        Arguments:
        y : tf.Tensor of shape (batch_size, summary_dim) -- the conditional data of interest y = sum(y)

        ----------

        Output:
        x_mean   : tf.Tensor of shape (batch_size, theta_dim) -- the mean of the reconstructed parameters
        z_var    : tf.Tensor of shape (batch_size, z_dim) -- the var of the reconstructed parameters

        """

        # Compute summary, if summary net has been given
        if self.summary_net is not None:
            y = self.summary_net(y)

        # Infer
        x = self.inference_net(y)
        theta_means, theta_vars = tf.split(x, 2, axis=-1)
        theta_vars = tf.nn.softplus(theta_vars)

        return theta_means, theta_vars

    def sample(self, y, n_samples, to_numpy=False, training=False):
        """
        Produces samples from the approximate Gaussian dsitro yielded by the inference net.
        """

        if self.summary_net is not None:
            y = self.summary_net(y, training=training)

        thetas = self.inference_net(y)
        theta_means, theta_vars = tf.split(thetas, 2, axis=-1)
        theta_vars = tf.nn.softplus(theta_vars)
        samples = tf.random_normal(shape=(n_samples, theta_means.shape[0], theta_means.shape[1]),
                                   mean=theta_means, stddev=tf.sqrt(theta_vars))

        if to_numpy:
            return samples.numpy()
        return samples



