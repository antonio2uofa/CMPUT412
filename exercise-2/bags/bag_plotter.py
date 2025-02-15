import rosbag
import math
import matplotlib.pyplot as plt

# Function to calculate robot path with a given angle exaggeration factor


def calculate_path(angle_exaggeration_factor):
    bag = rosbag.Bag('move.bag')
    linear_v, angular_v = [0], [0]
    theta = [0]
    x = [0]
    y = [0]
    index = 0
    previous_time = None

    for topic, msg, t in bag.read_messages(topics=['/csc22927/wheels_driver_node/wheels_cmd', 'numbers']):
        # Calculate time step
        current_time = t.to_sec()
        if previous_time is None:
            time_step = 0
        else:
            time_step = current_time - previous_time
        previous_time = current_time

        # Compute linear and angular velocities
        linear_v.append((msg.vel_right + msg.vel_left) / 2)
        angular_v.append((msg.vel_right - msg.vel_left) / 2)

        # Exaggerate the angle change
        theta.append(theta[index] + angular_v[index] *
                     time_step * angle_exaggeration_factor)

        # Update position
        x.append(x[-1] + linear_v[index] *
                 math.cos(theta[index + 1]) * time_step)
        y.append(y[-1] + linear_v[index] *
                 math.sin(theta[index + 1]) * time_step)

        index += 1

    bag.close()
    return x, y


# Create a figure with four subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Angle exaggeration factors
factors = [1.0, 2.0, 3.0, 5.0]

# Generate and plot paths for each factor
for i, factor in enumerate(factors):
    x, y = calculate_path(factor)
    row, col = divmod(i, 2)
    axes[row, col].plot(x, y, marker='o')
    axes[row, col].set_title(f'Angle Exaggeration Factor: {factor}')
    axes[row, col].set_xlabel('X')
    axes[row, col].set_ylabel('Y')

# Adjust layout and save the figure as a PDF
plt.tight_layout()
plt.savefig('exaggerated_robot_paths.pdf')
plt.show()
