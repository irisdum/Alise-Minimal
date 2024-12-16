from dataclasses import dataclass

from torch import Tensor, nn


@dataclass
class MLPDecoderConfig:
    """

    Parameters
    ----------
    inplanes : input number of features
    d_hidden : hidden number of features
    planes : output number of features
    """

    inplanes: int
    d_hidden: int
    planes: int


class MLPDecoder(nn.Module):
    """
    a simple MLP
    """

    def __init__(self, config: MLPDecoderConfig):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(config.inplanes, config.d_hidden),
            nn.ReLU(),
            nn.Linear(config.d_hidden, config.planes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)
