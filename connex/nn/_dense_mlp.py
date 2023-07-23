from collections.abc import Callable, Mapping
from typing import Any, Optional, Union

import jax.nn as jnn
import jax.random as jr
import numpy as np

from .._network import NeuralNetwork
from ._utils import _identity


class DenseMLP(NeuralNetwork):
    """
    A "Dense Multi-Layer Perceptron". Like a standard MLP, but
    every neuron is connected to every other neuron in all later layers, rather
    than only the next layer. That is, each layer uses the outputs from _all_ previous
    layers, not just the most recent one, in a similar manner to DenseNet.

    ??? cite

        [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
        ```bibtex
        @article{ba2016layer,
            author={Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger},  # noqa: E501
            title={Densely Connected Convolutional Networks},
            year={2016},
            journal={arXiv:1608.06993},
        }
        ```
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int,
        depth: int,
        hidden_activation: Callable = jnn.gelu,
        output_transformation: Callable = _identity,
        dropout_p: Union[float, Mapping[Any, float]] = 0.0,
        use_topo_norm: bool = False,
        use_topo_self_attention: bool = False,
        use_neuron_self_attention: bool = False,
        use_adaptive_activations: bool = False,
        *,
        key: Optional[jr.PRNGKey] = None,
    ):
        """**Arguments**:

        - `input_size`: The number of neurons in the input layer.
        - `output_size`: The number of neurons in the output layer.
        - `width`: The number of neurons in each hidden layer.
        - `depth`: The number of hidden layers.
        - `hidden_activation`: The activation function applied element-wise to
            the hidden (i.e. non-input, non-output) neurons. It can itself be a
            trainable `equinox Module`.
        - `output_transformation`: The transformation applied group-wise to the
            output neurons, e.g. `jax.nn.softmax`. It can itself be a trainable
            `equinox.Module`.
        - `dropout_p`: Dropout probability. If a single `float`, the same dropout
            probability will be applied to all hidden neurons.
            If a `Mapping[Any, float]`, `dropout_p[i]` refers to the dropout
            probability of neuron `i`. All neurons default to zero unless otherwise
            specified. Note that this allows dropout to be applied to input and output
            neurons as well.
        - `use_topo_norm`: A `bool` indicating whether to apply a topological batch-
            version of Layer Norm,

            ??? cite

                [Layer Normalization](https://arxiv.org/abs/1607.06450)
                ```bibtex
                @article{ba2016layer,
                    author={Jimmy Lei Ba, Jamie Ryan Kriso, Geoffrey E. Hinton},
                    title={Layer Normalization},
                    year={2016},
                    journal={arXiv:1607.06450},
                }
                ```

            where the collective inputs of each topological batch are standardized
            (made to have mean 0 and variance 1), with learnable elementwise-affine
            parameters `gamma` and `beta`.
        - `use_topo_self_attention`: A `bool` indicating whether to apply
            (single-headed) self-attention to each topological batch's collective
            inputs.

            ??? cite

                [Attention is All You Need](https://arxiv.org/abs/1706.03762)
                ```bibtex
                @inproceedings{vaswani2017attention,
                    author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and
                            Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and
                            Kaiser, {\L}ukasz and Polosukhin, Illia},
                    booktitle={Advances in Neural Information Processing Systems},
                    publisher={Curran Associates, Inc.},
                    title={Attention is All You Need},
                    volume={30},
                    year={2017}
                }
                ```

        - `use_neuron_self_attention`: A `bool` indicating whether to apply neuron-wise
            self-attention, where each neuron applies (single-headed) self-attention to
            its inputs. If both `_use_neuron_self_attention` and `use_neuron_norm` are
            `True`, normalization is applied before self-attention.

            !!! warning

                Neuron-level self-attention will use significantly more memory than
                than topo-level self-attention.

        - `use_adaptive_activations`: A bool indicating whether to use neuron-wise
            adaptive activations, where all hidden activations transform as
            `σ(x) -> a * σ(b * x)`, where `a`, `b` are trainable scalar parameters
            unique to each neuron.

            ??? cite

                [Locally adaptive activation functions with slope recovery term for
                 deep and physics-informed neural networks](https://arxiv.org/abs/1909.12228)  # noqa: E501
                ```bibtex
                @article{Jagtap_2020,
                    author={Ameya D. Jagtap, Kenji Kawaguchi, George Em Karniadakis},
                    title={Locally adaptive activation functions with slope recovery
                           term for deep and physics-informed neural networks},
                    year={2020},
                    publisher={The Royal Society},
                    journal={Proceedings of the Royal Society A: Mathematical, Physical
                    and Engineering Sciences},
                }
                ```
        - `key`: The `PRNGKey` used to initialize parameters. Optional, keyword-only
            argument. Defaults to `jax.random.PRNGKey(0)`.
        """
        key = key if key is not None else jr.PRNGKey(0)
        num_neurons = width * depth + input_size + output_size
        input_neurons = np.arange(input_size, dtype=int)
        output_neurons_start = num_neurons - output_size
        output_neurons = np.arange(output_size, dtype=int) + output_neurons_start
        adjacency_dict = {}
        layer_sizes = [input_size] + ([width] * depth) + [output_size]
        neuron = 0
        for layer_size in layer_sizes[:-1]:
            row_idx = range(neuron, neuron + layer_size)
            col_idx = range(neuron + layer_size, num_neurons)
            for r in row_idx:
                adjacency_dict[r] = list(col_idx)
            neuron += layer_size
        topo_sort = list(range(num_neurons))

        super().__init__(
            adjacency_dict,
            input_neurons,
            output_neurons,
            hidden_activation,
            output_transformation,
            dropout_p,
            use_topo_norm,
            use_topo_self_attention,
            use_neuron_self_attention,
            use_adaptive_activations,
            topo_sort=topo_sort,
            key=key,
        )
