"""Classes and utilities for generating gate configurations."""

from pathlib import Path
from typing import Tuple

import numpy as np


class GateGenerator:
    """Class to generate gate configurations for a drone racing course."""

    def __init__(
        self,
        gate_distance_range: np.ndarray = np.array([0.1, 1.0]),
        gate_angle_range: np.ndarray = np.array([-np.pi, np.pi]),
        gate_yaw_diff_range: np.ndarray = np.array([-np.pi / 10, np.pi / 10]),
        gate_heights: np.ndarray = np.array([0.525, 1.0]),
        gate_x_bounds: np.ndarray = np.array([-10.0, 10.0]),
        gate_y_bounds: np.ndarray = np.array([-10.0, 10.0]),
        num_gates: int = 5,
        edge_size: float = 0.525,
    ):
        """Initialize the gate generator.

        Args:
            gate_distance_range: Range of distances between gates.
            gate_angle_range: Range of angles between gates.
            gate_yaw_diff_range: Range of yaw differences between gates.
            gate_heights: Heights of the gates.
            gate_x_bounds: Bounds of the x-axis.
            gate_y_bounds: Bounds of the y-axis.
            num_gates: Number of gates.
            edge_size: Minimum distance between gates.
        """
        self.gate_distance_range = gate_distance_range
        self.num_gates = num_gates
        self.gate_angle_range = gate_angle_range
        self.gate_yaw_diff_range = gate_yaw_diff_range
        self.gate_heights = gate_heights
        self.gate_x_bounds = gate_x_bounds
        self.gate_y_bounds = gate_y_bounds
        self.edge_size = edge_size

    def generate_gates(
        self, initial_position: np.ndarray = np.zeros(3), initial_yaw: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate gate configurations for a drone racing course."""
        previous_gate_position = initial_position
        previous_gate_yaw = initial_yaw
        gate_positions = np.zeros((self.num_gates, 3))
        gate_yaws = np.zeros(self.num_gates)
        for i in range(self.num_gates):
            while True:
                gate_position, gate_yaw = self.sample_gate(previous_gate_position, previous_gate_yaw)
                print(gate_position)
                if self.is_in_bounds(gate_position) and self.do_not_collide(gate_positions, gate_position):
                    break

            gate_positions[i] = gate_position
            gate_yaws[i] = gate_yaw

            previous_gate_position = gate_position
            previous_gate_yaw = gate_yaw
        return gate_positions, gate_yaws

    def do_not_collide(self, gates_positions: np.ndarray, new_gate_position: np.ndarray) -> bool:
        """Check if the gate does not collide with other gates."""
        return np.all(np.linalg.norm(gates_positions[:, :2] - new_gate_position[:2], axis=1) > self.edge_size)

    def is_in_bounds(self, gate_position: np.ndarray) -> bool:
        """Check if the gate is within the bounds of the course."""
        return (
            gate_position[0] > self.gate_x_bounds[0]
            and gate_position[0] < self.gate_x_bounds[1]
            and gate_position[1] > self.gate_y_bounds[0]
            and gate_position[1] < self.gate_y_bounds[1]
        )

    def sample_gate(self, previous_gate_position: np.ndarray, previous_gate_yaw: float) -> np.ndarray:
        """Sample gate configurations for a drone racing course."""
        gate_distance = np.random.uniform(
            self.gate_distance_range[0],
            self.gate_distance_range[1],
        )
        gate_angle = np.random.uniform(
            self.gate_angle_range[0],
            self.gate_angle_range[1],
        )
        gate_yaw_diff = np.random.uniform(
            self.gate_yaw_diff_range[0],
            self.gate_yaw_diff_range[1],
        )
        new_gate_position = np.array(
            [
                previous_gate_position[0] + gate_distance * np.cos(previous_gate_yaw + np.pi / 2 + gate_angle),
                previous_gate_position[1] + gate_distance * np.sin(previous_gate_yaw + np.pi / 2 + gate_angle),
                np.random.choice(self.gate_heights),
            ]
        )
        new_gate_yaw = previous_gate_yaw + gate_yaw_diff
        return new_gate_position, new_gate_yaw

    def plot_gates(self, gate_positions: np.ndarray, gate_yaws: np.ndarray):
        """Plot the gate configurations for a drone racing course."""
        import matplotlib.pyplot as plt

        print("Plotting the gates")
        fig, ax = plt.subplots()
        for i, gate_position in enumerate(gate_positions):
            ax.text(gate_position[0], gate_position[1], str(i))
            ax.plot(
                [
                    gate_position[0] - self.edge_size / 2 * np.cos(gate_yaws[i]),
                    gate_position[0] + self.edge_size / 2 * np.cos(gate_yaws[i]),
                ],
                [
                    gate_position[1] - self.edge_size / 2 * np.sin(gate_yaws[i]),
                    gate_position[1] + self.edge_size / 2 * np.sin(gate_yaws[i]),
                ],
            )
        ax.set_aspect("equal")
        ax.set_xlim(self.gate_x_bounds)
        ax.set_ylim(self.gate_y_bounds)

        # Write the plot to a file
        print("Saving the plot to gates.png")
        plt.savefig("gates.png")

    def generate_gates_config(self, template_file: str = "config/level/template0.yaml", output_file: str = "config/level/generated.yaml"):
        """Generate a YAML file for the gate configurations."""
        import yaml

        project_root = Path(__file__).resolve().parents[1]

        with open(project_root / template_file, "r") as file:
            template = yaml.safe_load(file)

        initial_x = np.random.uniform(self.gate_x_bounds[0] * 0.5, self.gate_x_bounds[1] * 0.5)
        initial_y = np.random.uniform(self.gate_y_bounds[0] * 0.5, self.gate_y_bounds[1] * 0.5)
        initial_z = 1.0
        initial_yaw = 0.0
        initial_position = [initial_x, initial_y, initial_z]

        gate_positions, gate_yaws = self.generate_gates(
            initial_position=np.array(initial_position),
            initial_yaw=initial_yaw,
        )

        gates = []
        for i, (gate_position, gate_yaw) in enumerate(zip(gate_positions, gate_yaws)):
            gates.append(
                {
                    "position": gate_position.tolist(),
                    "yaw": gate_yaw,
                    "gate_id": i + 1,
                }
            )

        template["quadrotor_config"]["gates"] = [self.generate_gate_state(**gate) for gate in gates]
        template["quadrotor_config"]["obstacles"] = [self.generate_obstacle_state() for _ in range(5)]
        template["quadrotor_config"]["init_state"] = [
            self.generate_full_state(position=initial_position, yaw=initial_yaw, gate_id=0)
        ] + [self.generate_full_state(**gate) for gate in gates[:-1]]

        with open(project_root / output_file, "w") as file:
            yaml.dump(template, file)

    def generate_gate_state(self, position: np.ndarray = np.zeros(3), yaw: float = 0.0, gate_id: int = 0):
        """Generate the state of a gate."""
        # TODO: Explain the state vector
        # x, y, z, r, p, y, type (0: `tall` obstacle, 1: `low` obstacle)
        return [
            position[0],
            position[1],
            0.0,
            0.0,
            0.0,
            yaw,
            0 if position[2] > 0.75 else 1,
        ]

    def generate_obstacle_state(self, position: np.ndarray = np.zeros(3)):
        """Generate the state of an obstacle."""
        # obstacles:
        # [  # x, y, z, r, p, y
        # [0.5, -1.5, 0, 0, 0, 0],
        # ]
        return [
            position[0],
            position[1],
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    def generate_full_state(self, position: np.ndarray = np.zeros(3), yaw: float = 0.0, gate_id: int = 0):
        """Generate the full state of the drone racing course."""
        return [
            position[0],
            0.0,
            position[1],
            0.0,
            position[2],
            0.0,
            np.zeros(2),
            yaw,
            np.zeros(3),
            gate_id,
        ]
