from pathlib import Path

from lsy_drone_racing.level_generator import Level, interpolate_levels


def test_interpolate_levels():
    """Test the interpolate levels."""
    level1_path = "config/level/level_line0.yaml"
    level2_path = "config/level/level_train0.yaml"

    project_root = Path(__file__).resolve().parents[1]

    level1 = Level.from_yaml(project_root / level1_path)
    level2 = Level.from_yaml(project_root / level2_path)



    import matplotlib.pyplot as plt

    interpolated_level = interpolate_levels(level1, level2, 0.9)
    interpolated_level.sa

    fig, ax = plt.subplots()
    interpolated_level.plot(ax)
    interpolated_level.plot(ax, plot_gates=True)
    plt.show()

if __name__ == "__main__":
    test_interpolate_levels()
