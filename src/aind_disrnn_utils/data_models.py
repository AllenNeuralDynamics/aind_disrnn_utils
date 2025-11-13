"""
Defines data models for input and output settings
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class disRNNInputSettings(BaseSettings, cli_parse_args=True):
    """
    Pydantic settings model for input arguments.
    """

    subject_ids: list[int] = Field(
        default_factory=list, description="subject ids"
    )
    n_steps: int = Field(default=3000, description="Number of training steps")
    n_warmup_steps: int = Field(
        default=1000, description="Number of noiseless training steps"
    )
    beta: float = Field(default=1e-2, description="hyperparameter")
    learning_rate: float = Field(
        default=1e-3, description="Learning rate for optimization"
    )
    ignore_policy: str = Field(
        default="exclude",
        description="Whether to include or exclude ignored trials",
    )
    num_latents: int = Field(default=5, description="Number of latents to use")
    update_net_n_units_per_layer: int = Field(
        default=16,
        description="Number of units each each layer of update network",
    )
    update_net_n_layers: int = Field(
        default=8, description="Number of layers in update network"
    )
    choice_net_n_units_per_layer: int = Field(
        default=4,
        description="Number of units in each layer of choice network",
    )
    choice_net_n_layers: int = Field(
        default=1, description="Number of layers in choice network"
    )
    activation: str = Field(
        default="leaky_relu", description="Activation function"
    )
    multisubject: bool = Field(
        default=False, description="Whether to fit a multisubject disRNN"
    )
    features: dict[str, str] = Field(
        default_factory=lambda: {
            "animal_response": "prev choice",
            "rewarded": "prev reward",
        },
        description="input features for the RNN. "
        + "Keys are column names of df_trials, values are labels",
    )


class disRNNOutputSettings(BaseSettings):
    """
    Pydantic settings model for input arguments.
    """

    training_time: float = Field(description="training time")
    likelihood: float = Field(description="evaluation set likelihood")
    num_sessions: int = Field(description="number of sessions in full dataset")
    num_trials: int = Field(description="number of trials")
