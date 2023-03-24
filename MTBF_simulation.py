import random
import math
# Define the number of machines to simulate
num_machines = 10

# Define the initial MTBF value for each machine
mtbf_values = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]

# Define the threshold for detecting a defective machine
defective_threshold = 650

# Simulate the performance of each machine
for i in range(num_machines):
    # Generate a random number between 0 and 1
    r = random.uniform(0, 1)
    # Calculate the time until the next failure based on the MTBF value
    time_until_failure = -mtbf_values[i] * math.log(1 - r)
    # Print the time until the next failure for each machine
    print("Machine {}: Time until next failure = {:.2f}".format(i+1, time_until_failure))
    # Update the MTBF value for the machine based on the time until the next failure
    mtbf_values[i] = (mtbf_values[i] + time_until_failure) / 2.0

# Find the machine with the lowest MTBF value
worst_machine_index = mtbf_values.index(min(mtbf_values))
# Print the index and MTBF value of the worst performing machine
print("Worst performing machine: Index = {}, MTBF = {:.2f}".format(worst_machine_index+1, mtbf_values[worst_machine_index]))

# Check if any machine has a defective MTBF value
if any(mtbf_value < defective_threshold for mtbf_value in mtbf_values):
    print("Defective machine detected!")
else:
    print("All machines are performing well.")
