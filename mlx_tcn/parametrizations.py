import types

import mlx.core as mx

_WEIGHT_NORM_EPS = 1e-8


def _canonical_dim(weight: mx.array, dim: int) -> int:
    """Normalize the requested dimension into the [0, ndim) range."""
    ndim = len(weight.shape)
    if ndim == 0:
        raise ValueError("Cannot apply weight_norm to a scalar parameter.")
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"Invalid dim={dim} for weight with {ndim} dimensions.")
    return dim


def _compute_weight(module, name: str, meta: dict) -> mx.array:
    """Reconstruct the normalized weight from (g, v)."""
    v = getattr(module, f"{name}_v")
    g = getattr(module, f"{name}_g")
    dim = meta["dim"]
    v_norm = mx.linalg.norm(v, axis=dim, keepdims=True)
    return g * (v / (v_norm + _WEIGHT_NORM_EPS))


def _ensure_weight_norm_forward_hook(module):
    """Wrap module.__call__ once to materialize normalized weights before forward."""
    if hasattr(module, "_weight_norm_original_call"):
        return

    module._weight_norm_original_call = module.__call__

    def _call_with_weight_norm(self, *args, **kwargs):
        for param_name, meta in self._weight_norm_params.items():
            normalized = _compute_weight(self, param_name, meta)
            setattr(self, param_name, normalized)
        return self._weight_norm_original_call(*args, **kwargs)

    module.__call__ = types.MethodType(_call_with_weight_norm, module)


def weight_norm(module, name: str = "weight", dim: int = 0):
    """
    Apply weight normalization to a module parameter in the spirit of
    ``torch.nn.utils.weight_norm``.

    Parameters
    ----------
    module:
        The MLX module (e.g. ``nn.Conv1d`` or ``nn.Linear``) whose parameter should be
        reparameterised.
    name:
        Name of the parameter to normalise. Defaults to ``"weight"``.
    dim:
        Dimension along which to compute the weight vector norm. Negative indices
        are supported and follow Python's indexing rules.

    Returns
    -------
    module:
        The same module instance, now tracking the additional ``{name}_g`` and
        ``{name}_v`` parameters.
    """
    if not hasattr(module, name):
        raise ValueError(f"Module {type(module).__name__} has no parameter '{name}'.")

    weight = getattr(module, name)
    if not isinstance(weight, mx.array):
        raise TypeError(
            f"Parameter '{name}' on {type(module).__name__} is not an mx.array."
        )

    dim = _canonical_dim(weight, dim)

    if hasattr(module, "_weight_norm_params") and name in module._weight_norm_params:
        raise ValueError(f"Weight norm already applied to parameter '{name}'.")

    v = mx.array(weight)
    v_norm = mx.linalg.norm(v, axis=dim, keepdims=True)
    g = mx.array(v_norm)

    setattr(module, f"{name}_v", v)
    setattr(module, f"{name}_g", g)

    if not hasattr(module, "_weight_norm_params"):
        module._weight_norm_params = {}
    module._weight_norm_params[name] = {"dim": dim}

    _ensure_weight_norm_forward_hook(module)

    # Materialise an initial normalised weight so attribute access stays valid.
    setattr(module, name, _compute_weight(module, name, module._weight_norm_params[name]))

    return module


def remove_weight_norm(module, name: str = "weight"):
    """
    Undo :func:`weight_norm` for the specified parameter, restoring the original
    trainable weight.
    """
    if not hasattr(module, "_weight_norm_params") or name not in module._weight_norm_params:
        raise ValueError(f"Weight normalization not applied to parameter '{name}'.")

    meta = module._weight_norm_params[name]

    weight = _compute_weight(module, name, meta)
    setattr(module, name, weight)

    delattr(module, f"{name}_v")
    delattr(module, f"{name}_g")
    del module._weight_norm_params[name]

    if not module._weight_norm_params:
        if hasattr(module, "_weight_norm_original_call"):
            module.__call__ = module._weight_norm_original_call
            delattr(module, "_weight_norm_original_call")
        delattr(module, "_weight_norm_params")

    return module
