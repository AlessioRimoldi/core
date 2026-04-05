"""JAX/Flax modules for the deployable policy pipeline.

At training time, these operate as pure JAX functions on JAX arrays.
For ONNX export, the ``export_onnx`` module converts params to PyTorch
tensors and re-wraps them in ``torch.nn.Module``\\ s for ``torch.onnx.export``.
"""
