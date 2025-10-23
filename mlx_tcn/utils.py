import math


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    """Replicate torch.nn.init.calculate_gain for MLX initializers."""
    nl = nonlinearity.lower()

    linear_like = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transposed1d",
        "conv_transposed2d",
        "conv_transposed3d",
        "sigmoid",
        "tanh",
    }

    if nl in linear_like:
        return 1.0
    if nl == "relu":
        return math.sqrt(2.0)
    if nl == "leaky_relu":
        negative_slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1 + negative_slope**2))
    if nl == "selu":
        return 3.0 / 4.0
    if nl == "glu":
        return math.sqrt(2.0)

    raise ValueError(
        f"Unsupported nonlinearity '{nonlinearity}'. "
        "Expected one of linear, (transpose) conv, sigmoid, tanh, relu, "
        "leaky_relu, selu, glu."
    )
