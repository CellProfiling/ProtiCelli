"""EDM noise scheduler for ProtiCelli.

Wraps ``diffusers.EDMEulerScheduler`` with a convenience factory function.
"""

from diffusers.schedulers import EDMEulerScheduler


def create_edm_scheduler(
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    num_train_timesteps: int = 1000,
    prediction_type: str = "epsilon",
) -> EDMEulerScheduler:
    """Create an EDM Euler scheduler.

    Parameters
    ----------
    sigma_min : float
        Minimum noise level.
    sigma_max : float
        Maximum noise level.
    sigma_data : float
        Standard deviation of the data distribution.
    num_train_timesteps : int
        Number of training timesteps.
    prediction_type : str
        Model prediction type (``"epsilon"`` or ``"sample"``).

    Returns
    -------
    EDMEulerScheduler
        Configured scheduler instance.
    """
    return EDMEulerScheduler(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        prediction_type=prediction_type,
        num_train_timesteps=num_train_timesteps,
    )
