from typing import Any, Dict, List
import torch
from auto_diff import *


class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, node_A: Node, node_B: Node, normalized_shape: List[int], eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        input_A, input_B = input_values
        if input_A.shape[-1] != input_B.shape[-2]:
            input_B = torch.transpose(input_B, 0, 1)
            input_B = torch.transpose(input_B, -1, -2)

        matmul_output = torch.matmul(input_A, input_B)

        eps = node.attrs["eps"]
        normalized_shape = node.attrs["normalized_shape"]
        dim = tuple([i for i in range(-len(normalized_shape), 0)])
        mean = matmul_output.mean(dim=dim, keepdim=True)
        var = matmul_output.var(dim=dim, unbiased=False, keepdim=True)
        return (matmul_output - mean) / torch.sqrt(var + eps)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
    
        normalized_shape = node.attrs["normalized_shape"]
        dim = tuple([i for i in range(-len(normalized_shape), 0)])
        eps = node.attrs["eps"]

        input_A, input_B = node.inputs
        y = matmul(input_A, input_B)
        y_input_mean = mean(y, dim=dim, keepdim=True)
        y_input_var = mean(
            power(sub(y, y_input_mean), 2), dim=dim, keepdim=True
        )
        y_input_std = sqrt(y_input_var + eps)
        N = 1
        for shape in normalized_shape:
            N *= shape
        grad_y = div(output_grad, y_input_std)
        grad_mean = div(
            sum_op(output_grad, dim=dim, keepdim=True), mul_by_const(y_input_std, N)
        )
        y_input_minus_mean = sub(y, y_input_mean)
        numerator = mul(
            y_input_minus_mean,
            sum_op(mul(output_grad, y_input_minus_mean), dim=dim, keepdim=True),
        )
        denominator = mul_by_const(power(y_input_std, 3.0), N)
        grad_var = div(numerator, denominator)
        grad_layernorm = sub(sub(grad_y, grad_mean), grad_var)
        B_T = transpose(input_B, -1, -2)
        A_T = transpose(input_A, -1, -2)
        return [matmul(grad_layernorm, B_T), matmul(A_T, grad_layernorm)]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(self, node_A: Node, node_B: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"dim": dim},
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        input_A, input_B = input_values
        if input_A.shape[-1] != input_B.shape[-2]:
            input_B = torch.transpose(input_B, 0, 1)
            input_B = torch.transpose(input_B, -1, -2)
        matmul_output = torch.matmul(input_A, input_B)
        input_max, _ = torch.max(matmul_output, dim=node.attrs["dim"], keepdim=True)
        input_exp = torch.exp(torch.sub(matmul_output, input_max))
        return input_exp / torch.sum(input_exp, dim=node.attrs["dim"], keepdim=True)


    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        """TODO: your code here"""
        input_A, input_B = node.inputs
        y = matmul(input_A, input_B)
        dim = node.attrs["dim"]
        soft_max = softmax(y, dim=dim)
        sum_softmax = sum_op(mul(soft_max, output_grad), dim=(dim,), keepdim=True)
        grad_softmax = mul(soft_max, output_grad - sum_softmax)
        B_T = transpose(input_B, -1, -2)
        A_T = transpose(input_A, -1, -2)
        return [matmul(grad_softmax, B_T), matmul(A_T, grad_softmax)]


# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
