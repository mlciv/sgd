from typing import List

import torch
from torch.nn import ParameterList, Parameter

class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.

    In addition, if `do_layer_norm=True` then apply layer normalization to each tensor
    before weighting.
    """

    def __init__(
        self,
        mixture_size: int,
        do_layer_norm: bool = False,
        initial_scalar_parameters: List[float] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise RuntimeError(
                "Length of initial_scalar_parameters {} differs "
                "from mixture_size {}".format(initial_scalar_parameters, mixture_size)
            )

        # Parameter list will not work for now : waiting for a patch erer, https://github.com/pytorch/pytorch/issues/36035
        # A workaround is just using parameter.
        # only tensor will not work, tensortype will not  show in the parameters list of this module. many device and copy issues
        self.scalar_parameters = Parameter(
            torch.FloatTensor(initial_scalar_parameters), requires_grad=trainable
        )
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    @staticmethod
    def tiny_value_of_dtype(dtype: torch.dtype):
        """
        Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
        issues such as division by zero.
        This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
        Only supports floating point dtypes.
        """
        if not dtype.is_floating_point:
            raise TypeError("Only supports floating point dtypes.")
        if dtype == torch.float or dtype == torch.double:
            return 1e-13
        elif dtype == torch.half:
            return 1e-4
        else:
            raise TypeError("Does not support dtype " + str(dtype))

    def forward(self, tensors: List[torch.Tensor], mask: torch.BoolTensor = None) -> torch.Tensor:
        """
        Compute a weighted average of the `tensors`.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.

        When `do_layer_norm=True`, the `mask` is required input.  If the `tensors` are
        dimensioned  `(dim_0, ..., dim_{n-1}, dim_n)`, then the `mask` is dimensioned
        `(dim_0, ..., dim_{n-1})`, as in the typical case with `tensors` of shape
        `(batch_size, timesteps, dim)` and `mask` of shape `(batch_size, timesteps)`.

        When `do_layer_norm=False` the `mask` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise RuntimeError(
                "{} tensors were passed, but the module was initialized to "
                "mix {} tensors.".format(len(tensors), self.mixture_size)
            )

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = (
                torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2) / num_elements_not_masked
            )
            return (tensor - mean) / torch.sqrt(variance + tiny_value_of_dtype(variance.dtype))

        normed_weights = torch.nn.functional.softmax(
            self.scalar_parameters, dim=0
        )
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)

        else:
            broadcast_mask = mask.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(
                    weight * _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked)
                )
            return self.gamma * sum(pieces)
