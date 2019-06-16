import unittest
import numpy as np
import tensorflow as tf
from models import CouplingNet, ConditionalInvertibleBlock, DeepConditionalModel
from losses import maximum_likelihood_loss


tf.enable_eager_execution()
tf.set_random_seed(42)


class InvertiblePermutationTest(unittest.TestCase):
    """Tests for all components of an invertible neural network."""

    def __init__(self, *args, **kwargs):
        super(InvertiblePermutationTest, self).__init__(*args, **kwargs)

        # Define the structure of a coupling block
        self.meta = {
            'n_units': [64, 64, 64, 64, 64],
            'activation': 'elu',
            'w_decay': 0.0001,
            'initializer': 'glorot_uniform'
        }
        self.x_dim_even = 6
        self.x_dim_odd = 5
        self.summary_dim = 32
        self.batch_size = 64
        self.z_sample_size = 100

        # Classes to test
        self.coupling_net_even = CouplingNet(self.meta, self.x_dim_even // 2)
        self.cINN_even = ConditionalInvertibleBlock(self.meta, self.x_dim_even, permute=True)
        self.coupling_net_odd = CouplingNet(self.meta, self.x_dim_odd // 2)
        self.cINN_odd = ConditionalInvertibleBlock(self.meta, self.x_dim_odd, permute=True)
        self.dINN_even = DeepConditionalModel(self.meta, 10, self.x_dim_even, permute=True)
        self.dINN_odd = DeepConditionalModel(self.meta, 10, self.x_dim_odd, permute=True)

        # Inputs to test on
        self.x_single_even = tf.random_normal((1, self.x_dim_even))
        self.x_single_odd = tf.random_normal((1, self.x_dim_odd))
        self.y_single = tf.random_normal((1, self.summary_dim))
        self.x_batch_even = tf.random_normal((self.batch_size, self.x_dim_even))
        self.x_batch_odd = tf.random_normal((self.batch_size, self.x_dim_odd))
        self.y_batch = tf.random_normal((self.batch_size, self.summary_dim))

    def test_shapes_coupling_even(self):
        """Tests the integrity of the coupling block on an even input."""

        out_single = self.coupling_net_even(self.x_single_even, self.y_single)
        out_batch = self.coupling_net_even(self.x_batch_even, self.y_batch)

        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in CouplingNet')
        self.assertEqual(out_single.shape[1], self.x_dim_even//2,
                         'Input/Output shape mismatch on single instance in CouplingNet')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in CouplingNet')
        self.assertEqual(out_batch.shape[1], self.x_dim_even // 2,
                         'Input/Output shape mismatch on a batch in CouplingNet')

    def test_shapes_coupling_out(self):
        """Tests the integrity of the coupling block on an odd input."""

        out_single = self.coupling_net_odd(self.x_single_odd, self.y_single)
        out_batch = self.coupling_net_odd(self.x_batch_odd, self.y_batch)

        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in CouplingNet')
        self.assertEqual(out_single.shape[1], self.x_dim_odd//2,
                         'Input/Output shape mismatch on single instance in CouplingNet')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in CouplingNet')
        self.assertEqual(out_batch.shape[1], self.x_dim_odd // 2,
                         'Input/Output shape mismatch on a batch in CouplingNet')

    def test_shapes_cinn_even(self):
        """Tests the integrity of the invertible block on an even input"""

        out_single, out_single_J = self.cINN_even(self.x_single_even, self.y_single)
        out_batch, out_batch_J = self.cINN_even(self.x_batch_even, self.y_batch)

        # Test shapes of output
        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_single.shape[1], self.x_dim_even,
                         'Input/Output shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in ConditionalInvertibleBlock')
        self.assertEqual(out_batch.shape[1], self.x_dim_even,
                         'Input/Output shape mismatch on a batch in ConditionalInvertibleBlock')

        # Test shapes of Jacobian
        self.assertEqual(out_single_J.shape[0], 1,
                         'Batch shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_batch_J.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in ConditionalInvertibleBlock')

        # Test equal batch sizes
        self.assertEqual(out_single.shape[0], out_single_J.shape[0],
                         'Batch shape mismatch between J and output in ConditonalInvertibleBlock')
        self.assertEqual(out_batch.shape[0], out_batch_J.shape[0],
                         'Batch shape mismatch between J and output in ConditonalInvertibleBlock')

    def test_shapes_cinn_odd(self):
        """Tests the integrity of the invertible block on an even input"""

        out_single, out_single_J = self.cINN_odd(self.x_single_odd, self.y_single)
        out_batch, out_batch_J = self.cINN_odd(self.x_batch_odd, self.y_batch)

        # Test shapes of output
        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_single.shape[1], self.x_dim_odd,
                         'Input/Output shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in ConditionalInvertibleBlock')
        self.assertEqual(out_batch.shape[1], self.x_dim_odd,
                         'Input/Output shape mismatch on a batch in ConditionalInvertibleBlock')

        # Test shapes of Jacobian
        self.assertEqual(out_single_J.shape[0], 1,
                         'Batch shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_batch_J.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in ConditionalInvertibleBlock')

        # Test equal batch sizes
        self.assertEqual(out_single.shape[0], out_single_J.shape[0],
                         'Batch shape mismatch between J and output in ConditonalInvertibleBlock')
        self.assertEqual(out_batch.shape[0], out_batch_J.shape[0],
                         'Batch shape mismatch between J and output in ConditonalInvertibleBlock')

    def test_shapes_dinn_even(self):
        """Test the integrity of the deep invertible chain on even inputs."""

        out_single, out_single_J = self.dINN_even(self.x_single_even, self.y_single)
        out_batch, out_batch_J = self.dINN_even(self.x_batch_even, self.y_batch)

        # Test shapes of output
        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_single.shape[1], self.x_dim_even,
                         'Input/Output shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[1], self.x_dim_even,
                         'Input/Output shape mismatch on a batch in DeepInvertibleModel')

        # Test shapes of Jacobian
        self.assertEqual(out_single_J.shape[0], 1,
                         'Batch shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_batch_J.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in DeepInvertibleModel')

        # Test equal batch sizes
        self.assertEqual(out_single.shape[0], out_single_J.shape[0],
                         'Batch shape mismatch between J and output in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[0], out_batch_J.shape[0],
                         'Batch shape mismatch between J and output in DeepInvertibleModel')

    def test_shapes_dinn_odd(self):
        """Test the integrity of the deep invertible chain on odd inputs."""

        out_single, out_single_J = self.dINN_odd(self.x_single_odd, self.y_single)
        out_batch, out_batch_J = self.dINN_odd(self.x_batch_odd, self.y_batch)

        # Test shapes of output
        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_single.shape[1], self.x_dim_odd,
                         'Input/Output shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[1], self.x_dim_odd,
                         'Input/Output shape mismatch on a batch in DeepInvertibleModel')

        # Test shapes of Jacobian
        self.assertEqual(out_single_J.shape[0], 1,
                         'Batch shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_batch_J.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in DeepInvertibleModel')

        # Test equal batch sizes
        self.assertEqual(out_single.shape[0], out_single_J.shape[0],
                         'Batch shape mismatch between J and output in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[0], out_batch_J.shape[0],
                         'Batch shape mismatch between J and output in DeepInvertibleModel')

    def test_cinn_invertible_even(self):
        """Tests invertibility of the cINN block on even inputs."""

        out_single, out_single_J = self.cINN_even(self.x_single_even, self.y_single)
        out_batch, out_batch_J = self.cINN_even(self.x_batch_even, self.y_batch)

        rec_single = self.cINN_even(out_single, self.y_single, inverse=True)
        rec_batch = self.cINN_even(out_batch, self.y_batch, inverse=True)

        self.assertTrue(np.allclose(self.x_single_even.numpy(), rec_single.numpy(), atol=1e-6),
                        'Could not invert ConditionalInvertibleBlock on a single instance with even z')
        self.assertTrue(np.allclose(self.x_batch_even.numpy(), rec_batch.numpy(), atol=1e-6),
                        'Could not invert ConditionalInvertibleBlock on batch instance with even z')

    def test_dinn_invertible_even(self):
        """Tests invertibility of the deep invertible chain on even inputs."""

        out_single, out_single_J = self.dINN_even(self.x_single_even, self.y_single)
        out_batch, out_batch_J = self.dINN_even(self.x_batch_even, self.y_batch)

        rec_single = self.dINN_even(out_single, self.y_single, inverse=True)
        rec_batch = self.dINN_even(out_batch, self.y_batch, inverse=True)

        self.assertTrue(np.allclose(self.x_single_even.numpy(), rec_single.numpy(), atol=1e-4),
                        'Could not invert DeepInvertibleModel on a single instance with even z')
        self.assertTrue(np.allclose(self.x_batch_even.numpy(), rec_batch.numpy(), atol=1e-4),
                        'Could not invert DeepInvertibleModel on batch instance with even z')

    def test_cinn_invertible_odd(self):
        """Tests invertibility of the cINN block on odd inputs."""

        out_single, out_single_J = self.cINN_odd(self.x_single_odd, self.y_single)
        out_batch, out_batch_J = self.cINN_odd(self.x_batch_odd, self.y_batch)

        rec_single = self.cINN_odd(out_single, self.y_single, inverse=True)
        rec_batch = self.cINN_odd(out_batch, self.y_batch, inverse=True)

        self.assertTrue(np.allclose(self.x_single_odd.numpy(), rec_single.numpy(), atol=1e-6),
                        'Could not invert ConditionalInvertibleBlock on a single instance with odd z')
        self.assertTrue(np.allclose(self.x_batch_odd.numpy(), rec_batch.numpy(), atol=1e-6),
                        'Could not invert ConditionalInvertibleBlock on batch instance with odd z')

    def test_dinn_invertible_odd(self):
        """Tests invertibility of the deep invertible chain on odd inputs."""

        out_single, out_single_J = self.dINN_odd(self.x_single_odd, self.y_single)
        out_batch, out_batch_J = self.dINN_odd(self.x_batch_odd, self.y_batch)

        rec_single = self.dINN_odd(out_single, self.y_single, inverse=True)
        rec_batch = self.dINN_odd(out_batch, self.y_batch, inverse=True)

        self.assertTrue(np.allclose(self.x_single_odd.numpy(), rec_single.numpy(), atol=1e-4),
                        'Could not invert DeepInvertibleModel on a single instance with odd z')
        self.assertTrue(np.allclose(self.x_batch_odd.numpy(), rec_batch.numpy(), atol=1e-4),
                        'Could not invert DeepInvertibleModel on batch instance with odd z')

    def test_diin_sampling_even(self):
        """Test integrity of sampling operation in the invertible chain."""

        # Sample single instance
        samples_single = self.dINN_even.sample(self.y_single, self.z_sample_size)

        self.assertEqual(samples_single.shape[0], self.z_sample_size, "Sample shape mismatch in "
                                                               "DeepInvertibleModel on even single outputs.")
        self.assertEqual(samples_single.shape[1], self.x_dim_even, "Sample shape mismatch in "
                                                               "DeepInvertibleModel on even single inputs.")

        # Sample batch
        samples_batch = self.dINN_even.sample(self.y_batch, self.z_sample_size)
        self.assertEqual(samples_batch.shape[0], self.z_sample_size, "Sample shape mismatch in "
                                                                      "DeepInvertibleModel on even batch outputs.")
        self.assertEqual(samples_batch.shape[1], self.batch_size, "Sample shape mismatch in "
                                                                   "DeepInvertibleModel on even batch inputs.")
        self.assertEqual(samples_batch.shape[2], self.x_dim_even, "Sample shape mismatch in "
                                                                  "DeepInvertibleModel on even batch inputs.")

    def test_diin_sampling_odd(self):
        """Test integrity of sampling operation in the invertible chain."""

        samples = self.dINN_odd.sample(self.y_single, self.z_sample_size)
        self.assertEqual(samples.shape[0], self.z_sample_size,
                         "Sample shape mismatch in DeepInvertibleModel on odd inputs.")
        self.assertEqual(samples.shape[1], self.x_dim_odd,
                         "Sample shape mismatch in DeepInvertibleModel on odd inputs")

        # Sample batch
        samples_batch = self.dINN_odd.sample(self.y_batch, self.z_sample_size,)
        self.assertEqual(samples_batch.shape[0], self.z_sample_size, "Sample shape mismatch in "
                                                                     "DeepInvertibleModel on odd batch outputs.")
        self.assertEqual(samples_batch.shape[1], self.batch_size, "Sample shape mismatch in "
                                                                  "DeepInvertibleModel on odd batch inputs.")
        self.assertEqual(samples_batch.shape[2], self.x_dim_odd, "Sample shape mismatch in "
                                                                  "DeepInvertibleModel on odd batch inputs.")

    def test_maximum_likelihood_loss_odd(self):
        """Tests the integrity of the maximum likelihood loss on odd inputs."""

        out_single, out_single_J = self.dINN_odd(self.x_single_odd, self.y_single)
        out_batch, out_batch_J = self.dINN_odd(self.x_batch_odd, self.y_batch)

        single_ml = maximum_likelihood_loss(out_single, out_single_J)
        batch_ml = maximum_likelihood_loss(out_batch, out_batch_J)

        self.assertEqual(single_ml.shape, tf.TensorShape([]), "Sample shape mismatch in ML loss in "
                                                                "DeepInvertibleModel on odd single inputs.")
        self.assertEqual(batch_ml.shape, tf.TensorShape([]), "Sample shape mismatch in ML loss in "
                                                                "DeepInvertibleModel on odd batch inputs.")

    def test_maximum_likelihood_loss_even(self):
        """Tests the integrity of the maximum likelihood loss on even inputs."""

        out_single, out_single_J = self.dINN_even(self.x_single_even, self.y_single)
        out_batch, out_batch_J = self.dINN_even(self.x_batch_even, self.y_batch)

        single_ml = maximum_likelihood_loss(out_single, out_single_J)
        batch_ml = maximum_likelihood_loss(out_batch, out_batch_J)

        self.assertEqual(single_ml.shape, tf.TensorShape([]), "Sample shape mismatch in ML loss in "
                                                                "DeepInvertibleModel on even single inputs.")
        self.assertEqual(batch_ml.shape, tf.TensorShape([]), "Sample shape mismatch in ML loss in "
                                                                "DeepInvertibleModel on even batch inputs.")


class InvertibleTest(unittest.TestCase):
    """Tests for all components of an invertible neural network."""

    def __init__(self, *args, **kwargs):
        super(InvertibleTest, self).__init__(*args, **kwargs)

        # Define the structure of a coupling block
        self.meta = {
            'n_units': [64, 64, 64, 64, 64],
            'activation': 'elu',
            'w_decay': 0.0001,
            'initializer': 'glorot_uniform'
        }
        self.x_dim_even = 6
        self.x_dim_odd = 5
        self.summary_dim = 32
        self.batch_size = 64
        self.z_sample_size = 100

        # Classes to test
        self.coupling_net_even = CouplingNet(self.meta, self.x_dim_even // 2)
        self.cINN_even = ConditionalInvertibleBlock(self.meta, self.x_dim_even, permute=False)
        self.coupling_net_odd = CouplingNet(self.meta, self.x_dim_odd // 2)
        self.cINN_odd = ConditionalInvertibleBlock(self.meta, self.x_dim_odd,permute=False)
        self.dINN_even = DeepConditionalModel(self.meta, 10, self.x_dim_even, permute=False)
        self.dINN_odd = DeepConditionalModel(self.meta, 10, self.x_dim_odd, permute=False)

        # Inputs to test on
        self.x_single_even = tf.random_normal((1, self.x_dim_even))
        self.x_single_odd = tf.random_normal((1, self.x_dim_odd))
        self.y_single = tf.random_normal((1, self.summary_dim))
        self.x_batch_even = tf.random_normal((self.batch_size, self.x_dim_even))
        self.x_batch_odd = tf.random_normal((self.batch_size, self.x_dim_odd))
        self.y_batch = tf.random_normal((self.batch_size, self.summary_dim))

    def test_shapes_coupling_even(self):
        """Tests the integrity of the coupling block on an even input."""

        out_single = self.coupling_net_even(self.x_single_even, self.y_single)
        out_batch = self.coupling_net_even(self.x_batch_even, self.y_batch)

        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in CouplingNet')
        self.assertEqual(out_single.shape[1], self.x_dim_even//2,
                         'Input/Output shape mismatch on single instance in CouplingNet')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in CouplingNet')
        self.assertEqual(out_batch.shape[1], self.x_dim_even // 2,
                         'Input/Output shape mismatch on a batch in CouplingNet')

    def test_shapes_coupling_out(self):
        """Tests the integrity of the coupling block on an odd input."""

        out_single = self.coupling_net_odd(self.x_single_odd, self.y_single)
        out_batch = self.coupling_net_odd(self.x_batch_odd, self.y_batch)

        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in CouplingNet')
        self.assertEqual(out_single.shape[1], self.x_dim_odd//2,
                         'Input/Output shape mismatch on single instance in CouplingNet')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in CouplingNet')
        self.assertEqual(out_batch.shape[1], self.x_dim_odd // 2,
                         'Input/Output shape mismatch on a batch in CouplingNet')

    def test_shapes_cinn_even(self):
        """Tests the integrity of the invertible block on an even input"""

        out_single, out_single_J = self.cINN_even(self.x_single_even, self.y_single)
        out_batch, out_batch_J = self.cINN_even(self.x_batch_even, self.y_batch)

        # Test shapes of output
        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_single.shape[1], self.x_dim_even,
                         'Input/Output shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in ConditionalInvertibleBlock')
        self.assertEqual(out_batch.shape[1], self.x_dim_even,
                         'Input/Output shape mismatch on a batch in ConditionalInvertibleBlock')

        # Test shapes of Jacobian
        self.assertEqual(out_single_J.shape[0], 1,
                         'Batch shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_batch_J.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in ConditionalInvertibleBlock')

        # Test equal batch sizes
        self.assertEqual(out_single.shape[0], out_single_J.shape[0],
                         'Batch shape mismatch between J and output in ConditonalInvertibleBlock')
        self.assertEqual(out_batch.shape[0], out_batch_J.shape[0],
                         'Batch shape mismatch between J and output in ConditonalInvertibleBlock')

    def test_shapes_cinn_odd(self):
        """Tests the integrity of the invertible block on an even input"""

        out_single, out_single_J = self.cINN_odd(self.x_single_odd, self.y_single)
        out_batch, out_batch_J = self.cINN_odd(self.x_batch_odd, self.y_batch)

        # Test shapes of output
        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_single.shape[1], self.x_dim_odd,
                         'Input/Output shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in ConditionalInvertibleBlock')
        self.assertEqual(out_batch.shape[1], self.x_dim_odd,
                         'Input/Output shape mismatch on a batch in ConditionalInvertibleBlock')

        # Test shapes of Jacobian
        self.assertEqual(out_single_J.shape[0], 1,
                         'Batch shape mismatch on single instance in ConditionalInvertibleBlock')
        self.assertEqual(out_batch_J.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in ConditionalInvertibleBlock')

        # Test equal batch sizes
        self.assertEqual(out_single.shape[0], out_single_J.shape[0],
                         'Batch shape mismatch between J and output in ConditonalInvertibleBlock')
        self.assertEqual(out_batch.shape[0], out_batch_J.shape[0],
                         'Batch shape mismatch between J and output in ConditonalInvertibleBlock')

    def test_shapes_dinn_even(self):
        """Test the integrity of the deep invertible chain on even inputs."""

        out_single, out_single_J = self.dINN_even(self.x_single_even, self.y_single)
        out_batch, out_batch_J = self.dINN_even(self.x_batch_even, self.y_batch)

        # Test shapes of output
        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_single.shape[1], self.x_dim_even,
                         'Input/Output shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[1], self.x_dim_even,
                         'Input/Output shape mismatch on a batch in DeepInvertibleModel')

        # Test shapes of Jacobian
        self.assertEqual(out_single_J.shape[0], 1,
                         'Batch shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_batch_J.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in DeepInvertibleModel')

        # Test equal batch sizes
        self.assertEqual(out_single.shape[0], out_single_J.shape[0],
                         'Batch shape mismatch between J and output in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[0], out_batch_J.shape[0],
                         'Batch shape mismatch between J and output in DeepInvertibleModel')

    def test_shapes_dinn_odd(self):
        """Test the integrity of the deep invertible chain on odd inputs."""

        out_single, out_single_J = self.dINN_odd(self.x_single_odd, self.y_single)
        out_batch, out_batch_J = self.dINN_odd(self.x_batch_odd, self.y_batch)

        # Test shapes of output
        self.assertEqual(out_single.shape[0], 1,
                         'Batch shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_single.shape[1], self.x_dim_odd,
                         'Input/Output shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[1], self.x_dim_odd,
                         'Input/Output shape mismatch on a batch in DeepInvertibleModel')

        # Test shapes of Jacobian
        self.assertEqual(out_single_J.shape[0], 1,
                         'Batch shape mismatch on single instance in DeepInvertibleModel')
        self.assertEqual(out_batch_J.shape[0], self.batch_size,
                         'Batch shape mismatch on a batch in DeepInvertibleModel')

        # Test equal batch sizes
        self.assertEqual(out_single.shape[0], out_single_J.shape[0],
                         'Batch shape mismatch between J and output in DeepInvertibleModel')
        self.assertEqual(out_batch.shape[0], out_batch_J.shape[0],
                         'Batch shape mismatch between J and output in DeepInvertibleModel')

    def test_cinn_invertible_even(self):
        """Tests invertibility of the cINN block on even inputs."""

        out_single, out_single_J = self.cINN_even(self.x_single_even, self.y_single)
        out_batch, out_batch_J = self.cINN_even(self.x_batch_even, self.y_batch)

        rec_single = self.cINN_even(out_single, self.y_single, inverse=True)
        rec_batch = self.cINN_even(out_batch, self.y_batch, inverse=True)

        self.assertTrue(np.allclose(self.x_single_even.numpy(), rec_single.numpy(), atol=1e-6),
                        'Could not invert ConditionalInvertibleBlock on a single instance with even z')
        self.assertTrue(np.allclose(self.x_batch_even.numpy(), rec_batch.numpy(), atol=1e-6),
                        'Could not invert ConditionalInvertibleBlock on batch instance with even z')

    def test_dinn_invertible_even(self):
        """Tests invertibility of the deep invertible chain on even inputs."""

        out_single, out_single_J = self.dINN_even(self.x_single_even, self.y_single)
        out_batch, out_batch_J = self.dINN_even(self.x_batch_even, self.y_batch)

        rec_single = self.dINN_even(out_single, self.y_single, inverse=True)
        rec_batch = self.dINN_even(out_batch, self.y_batch, inverse=True)

        self.assertTrue(np.allclose(self.x_single_even.numpy(), rec_single.numpy(), atol=1e-4),
                        'Could not invert DeepInvertibleModel on a single instance with even z')
        self.assertTrue(np.allclose(self.x_batch_even.numpy(), rec_batch.numpy(), atol=1e-4),
                        'Could not invert DeepInvertibleModel on batch instance with even z')

    def test_cinn_invertible_odd(self):
        """Tests invertibility of the cINN block on odd inputs."""

        out_single, out_single_J = self.cINN_odd(self.x_single_odd, self.y_single)
        out_batch, out_batch_J = self.cINN_odd(self.x_batch_odd, self.y_batch)

        rec_single = self.cINN_odd(out_single, self.y_single, inverse=True)
        rec_batch = self.cINN_odd(out_batch, self.y_batch, inverse=True)

        self.assertTrue(np.allclose(self.x_single_odd.numpy(), rec_single.numpy(), atol=1e-6),
                        'Could not invert ConditionalInvertibleBlock on a single instance with odd z')
        self.assertTrue(np.allclose(self.x_batch_odd.numpy(), rec_batch.numpy(), atol=1e-6),
                        'Could not invert ConditionalInvertibleBlock on batch instance with odd z')

    def test_dinn_invertible_odd(self):
        """Tests invertibility of the deep invertible chain on odd inputs."""

        out_single, out_single_J = self.dINN_odd(self.x_single_odd, self.y_single)
        out_batch, out_batch_J = self.dINN_odd(self.x_batch_odd, self.y_batch)

        rec_single = self.dINN_odd(out_single, self.y_single, inverse=True)
        rec_batch = self.dINN_odd(out_batch, self.y_batch, inverse=True)

        self.assertTrue(np.allclose(self.x_single_odd.numpy(), rec_single.numpy(), atol=1e-4),
                        'Could not invert DeepInvertibleModel on a single instance with odd z')
        self.assertTrue(np.allclose(self.x_batch_odd.numpy(), rec_batch.numpy(), atol=1e-4),
                        'Could not invert DeepInvertibleModel on batch instance with odd z')

    def test_diin_sampling_even(self):
        """Test integrity of sampling operation in the invertible chain."""

        # Sample single instance
        samples_single = self.dINN_even.sample(self.y_single, self.z_sample_size)

        self.assertEqual(samples_single.shape[0], self.z_sample_size, "Sample shape mismatch in "
                                                               "DeepInvertibleModel on even single outputs.")
        self.assertEqual(samples_single.shape[1], self.x_dim_even, "Sample shape mismatch in "
                                                               "DeepInvertibleModel on even single inputs.")

        # Sample batch
        samples_batch = self.dINN_even.sample(self.y_batch, self.z_sample_size)
        self.assertEqual(samples_batch.shape[0], self.z_sample_size, "Sample shape mismatch in "
                                                                      "DeepInvertibleModel on even batch outputs.")
        self.assertEqual(samples_batch.shape[1], self.batch_size, "Sample shape mismatch in "
                                                                   "DeepInvertibleModel on even batch inputs.")
        self.assertEqual(samples_batch.shape[2], self.x_dim_even, "Sample shape mismatch in "
                                                                  "DeepInvertibleModel on even batch inputs.")

    def test_diin_sampling_odd(self):
        """Test integrity of sampling operation in the invertible chain."""

        samples = self.dINN_odd.sample(self.y_single, self.z_sample_size)
        self.assertEqual(samples.shape[0], self.z_sample_size,
                         "Sample shape mismatch in DeepInvertibleModel on odd inputs.")
        self.assertEqual(samples.shape[1], self.x_dim_odd,
                         "Sample shape mismatch in DeepInvertibleModel on odd inputs")

        # Sample batch
        samples_batch = self.dINN_odd.sample(self.y_batch, self.z_sample_size)
        self.assertEqual(samples_batch.shape[0], self.z_sample_size, "Sample shape mismatch in "
                                                                     "DeepInvertibleModel on odd batch outputs.")
        self.assertEqual(samples_batch.shape[1], self.batch_size, "Sample shape mismatch in "
                                                                  "DeepInvertibleModel on odd batch inputs.")
        self.assertEqual(samples_batch.shape[2], self.x_dim_odd, "Sample shape mismatch in "
                                                                  "DeepInvertibleModel on odd batch inputs.")

    def test_maximum_likelihood_loss_odd(self):
        """Tests the integrity of the maximum likelihood loss on odd inputs."""

        out_single, out_single_J = self.dINN_odd(self.x_single_odd, self.y_single)
        out_batch, out_batch_J = self.dINN_odd(self.x_batch_odd, self.y_batch)

        single_ml = maximum_likelihood_loss(out_single, out_single_J)
        batch_ml = maximum_likelihood_loss(out_batch, out_batch_J)

        self.assertEqual(single_ml.shape, tf.TensorShape([]), "Sample shape mismatch in ML loss in "
                                                                "DeepInvertibleModel on odd single inputs.")
        self.assertEqual(batch_ml.shape, tf.TensorShape([]), "Sample shape mismatch in ML loss in "
                                                                "DeepInvertibleModel on odd batch inputs.")

    def test_maximum_likelihood_loss_even(self):
        """Tests the integrity of the maximum likelihood loss on even inputs."""

        out_single, out_single_J = self.dINN_even(self.x_single_even, self.y_single)
        out_batch, out_batch_J = self.dINN_even(self.x_batch_even, self.y_batch)

        single_ml = maximum_likelihood_loss(out_single, out_single_J)
        batch_ml = maximum_likelihood_loss(out_batch, out_batch_J)

        self.assertEqual(single_ml.shape, tf.TensorShape([]), "Sample shape mismatch in ML loss in "
                                                                "DeepInvertibleModel on even single inputs.")
        self.assertEqual(batch_ml.shape, tf.TensorShape([]), "Sample shape mismatch in ML loss in "
                                                                "DeepInvertibleModel on even batch inputs.")

if __name__ == '__main__':

    unittest.main()

