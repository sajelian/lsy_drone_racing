"""Test state estimation classes."""

import numpy as np

from lsy_drone_racing.state_estimation import LowPassSpeedEstimator, SpeedEstimator, UnityStateEstimator


def test_low_pass_speed_estimator():
    """Test LowPassSpeedEstimator class."""
    state_dim = 2
    measurement_dim = 1
    alpha = 0.5
    initial_state = np.array([0.0, 0.0])
    estimator = LowPassSpeedEstimator(state_dim, measurement_dim, initial_state, alpha)

    control_input = np.array([1.0])
    estimator.predict(control_input)

    measurement = np.array([1.0])
    estimator.update(measurement)

    assert np.allclose(estimator.state, np.array([0.5, 1.0]))

if __name__ == "__main__":
    test_low_pass_speed_estimator()
    print("All tests pass.")
