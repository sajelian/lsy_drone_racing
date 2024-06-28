"""Functions to test the gate generator module."""

import numpy as np

from lsy_drone_racing.gate_generator import GateGenerator


def test_gate_generator():
    """Test the gate generator."""
    gate_generator = GateGenerator(
        num_gates=5,
        edge_size=0.525,
        gate_distance_range=np.array([0.1, 2.0]),
        # gate_distance_range=np.array([0.1, 1.0]),
        # gate_angle_range=np.array([0, 0]),
        gate_angle_range=[-np.pi / 4, np.pi / 4],
        gate_yaw_diff_range=[-np.pi / 4, np.pi / 4],
        gate_heights=np.array([0.525, 1.0]),
        gate_x_bounds=np.array([-10.0, 10.0]),
        gate_y_bounds=np.array([-10.0, 10.0]),
    )
    # Check the observable variables
    gate_positions, gate_yaws = gate_generator.generate_gates()

    # Plot the gates
    gate_generator.plot_gates(gate_positions, gate_yaws)

    # Generate yaml file
    gate_generator.generate_gates_config()


if __name__ == "__main__":
    test_gate_generator()
