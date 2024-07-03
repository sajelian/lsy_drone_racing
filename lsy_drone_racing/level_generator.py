"""Classes and utilities for generating gate configurations."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml


@dataclass
class Gate:
    """Class to represent a gate in a drone racing circuit."""

    position: np.ndarray
    yaw: float
    gate_id: int

    @classmethod
    def from_config(cls, config_list: List, gate_id: int) -> "Gate":  # noqa: ANN102
        """Create a gate from a configuration list."""
        position = config_list[:2] + [1.0]  # TODO: Explain the state vector
        yaw = config_list[5]
        return cls(position=position, yaw=yaw, gate_id=gate_id)


@dataclass
class Obstacle:
    """Class to represent an obstacle in a drone racing circuit."""

    position: np.ndarray

    @classmethod
    def from_config(cls, config_list: List) -> "Obstacle":  # noqa: ANN102
        """Create an obstacle from a configuration list."""
        position = config_list[:2] + [1.0]  # TODO: Explain the state vector
        return cls(position=position)


class Level:
    """Class to represent a level in a drone racing circuit."""

    def __init__(self, gates: List[Gate], obstacles: List[Obstacle]):
        """Initialize the level."""
        self.gates = gates
        self.obstacles = obstacles

    def plot(self, ax=None, label: str = "") -> None:
        """Plot the level."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        for gate in self.gates:
            ax.text(gate.position[0] + 0.1, gate.position[1] - 0.1, f"G{gate.gate_id}{label}")
            ax.plot(
                [
                    gate.position[0] - 0.525 / 2 * np.cos(gate.yaw),
                    gate.position[0] + 0.525 / 2 * np.cos(gate.yaw),
                ],
                [
                    gate.position[1] - 0.525 / 2 * np.sin(gate.yaw),
                    gate.position[1] + 0.525 / 2 * np.sin(gate.yaw),
                ],
                "b",
            )

        for i, obstacle in enumerate(self.obstacles):
            ax.plot(obstacle.position[0] + 0.1, obstacle.position[1] - 0.1, "ro")
            ax.text(obstacle.position[0], obstacle.position[1], f"O{i}{label}")

        ax.set_aspect("equal")
        ax.set_xlim([-3.0, 3.0])
        ax.set_ylim([-3.0, 3.0])

        # Write the plot to a file
        return ax

    def save_yaml(
        self, template_file: str = "config/level/template0.yaml", output_file: str = "config/level/generated.yaml"
    ):
        """Generate a YAML file for the gate configurations."""
        import yaml

        project_root = Path(__file__).resolve().parents[1]

        with open(project_root / template_file, "r") as file:
            template = yaml.safe_load(file)

        initial_x = np.random.uniform(self.gate_x_bounds[0] * 0.5, self.gate_x_bounds[1] * 0.5)
        initial_y = np.random.uniform(self.gate_y_bounds[0] * 0.5, self.gate_y_bounds[1] * 0.5)
        initial_z = 0.4
        initial_yaw = 0.0
        initial_position = [initial_x, initial_y, initial_z]

        
        gates = []
        for i, (gate_position, gate_yaw) in enumerate(zip(self.gates.gates
            gates.append(
                {
                    "position": gate_position.tolist(),
                    "yaw": gate_yaw,
                    "gate_id": i + 1,
                }
            )

        for i, (gate_position, gate_yaw) in enumerate(zip(gate_positions, gate_yaws)):
            gates.append(
                {
                    "position": gate_position.tolist(),
                    "yaw": gate_yaw,
                    "gate_id": i + 1,
                }
            )

        template["quadrotor_config"]["gates"] = [self.generate_gate_state(**gate) for gate in gates]
        template["quadrotor_config"]["obstacles"] = [self.generate_obstacle_state() for _ in range(1)]
        template["quadrotor_config"]["init_state"] = [
            self.generate_full_state(position=initial_position, yaw=initial_yaw, gate_id=0)
        ] + [self.generate_full_state(**gate) for gate in gates[:-1]]

        with open(project_root / output_file, "w") as file:
            yaml.dump(template, file)
                

    @classmethod
    def from_yaml(cls, file_path: Path) -> "Level":  # noqa: ANN102
        """Create a level from a YAML file."""
        assert file_path.exists(), f"File {file_path} does not exist."
        with open(file_path, "r") as file:
            level = yaml.safe_load(file)
            gates_config = level["quadrotor_config"]["gates"]
            obstacles_config = level["quadrotor_config"]["obstacles"]
            gates = [Gate.from_config(gate_config, gate_id) for gate_id, gate_config in enumerate(gates_config)]
            obstacles = [Obstacle.from_config(obstacle_config) for obstacle_config in obstacles_config]
        return cls(gates=gates, obstacles=obstacles)

    def __repr__(self) -> str:
        """Return the string representation of the level."""
        return f"Level(gates={self.gates}, obstacles={self.obstacles})"

    def __str__(self) -> str:
        """Return the string representation of the level."""
        return f"Level(gates={self.gates}, obstacles={self.obstacles})"


class LevelGenerator:
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

    def generate_gates_config(
        self, template_file: str = "config/level/template0.yaml", output_file: str = "config/level/generated.yaml"
    ):
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
        template["quadrotor_config"]["obstacles"] = [self.generate_obstacle_state() for _ in range(1)]
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


def interpolate_levels(level1: Level, level2: Level, interpolation_factor: float = 0.5) -> Level:
    """Interpolate between two levels."""
    assert len(level1.gates) == len(level2.gates), "The number of gates must be the same for both levels."
    assert len(level1.obstacles) == len(level2.obstacles), "The number of obstacles must be the same for both levels."

    # Interpolate the gates
    interpolated_gates = []
    for gate1, gate2 in zip(level1.gates, level2.gates):
        new_gate = Gate(
            position=(
                interpolation_factor * np.array(gate1.position)
                + (1 - interpolation_factor) * np.array(gate2.position)
            ).tolist(),
            yaw=interpolation_factor * gate1.yaw + (1 - interpolation_factor) * gate2.yaw,
            gate_id=gate1.gate_id,
        )
        interpolated_gates.append(move_gate_outside_of_gates(new_gate, interpolated_gates))

    # Interpolate the obstacles
    interpolated_obstacles = []
    for obstacle1, obstacle2 in zip(level1.obstacles, level2.obstacles):
        obstacle = Obstacle(
            position=(
                interpolation_factor * np.array(obstacle1.position)
                + (1 - interpolation_factor) * np.array(obstacle2.position)
            ).tolist()
        )
        interpolated_obstacles.append(move_obstacle_outside_of_gates(obstacle, interpolated_gates))

    return Level(gates=interpolated_gates, obstacles=interpolated_obstacles)


def move_obstacle_outside_of_gates(obstacle: Obstacle, gates: List[Gate], gate_edge_size: float = 0.525) -> Obstacle:
    """Move the obstacle outside of the gates."""
    obstacle_position = np.array(obstacle.position[:2])
    for gate in gates:
        gate_position = np.array(gate.position[:2])
        if np.linalg.norm(obstacle_position - gate_position) < gate_edge_size / 2:
            obs_distance = np.linalg.norm(obstacle_position - gate_position)
            obstacle_position = (
                obstacle_position
                + (gate_edge_size / 2 - obs_distance) * (obstacle_position - gate_position) / obs_distance
            )
            obstacle.position[:2] = obstacle_position

    return obstacle

def move_gate_outside_of_gates(gate: Gate, gates: List[Gate], gate_edge_size: float = 0.525) -> Gate:
    """Move the gate outside of the gates."""
    gate_position = np.array(gate.position[:2])
    for other_gate in gates:
        other_gate_position = np.array(other_gate.position[:2])
        if np.linalg.norm(gate_position - other_gate_position) < gate_edge_size / 2:
            gate_distance = np.linalg.norm(gate_position - other_gate_position)
            gate_position = (
                gate_position
                + (gate_edge_size / 2 - gate_distance) * (gate_position - other_gate_position) / gate_distance
            )
            gate.position[:2] = gate_position

    return gate


def plot_levels(levels: List[Level]):
    """Plot the levels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for level in levels:
        ax = level.plot(ax=ax)

    plt.show()
