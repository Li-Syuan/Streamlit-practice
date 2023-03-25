import random
import numpy as np
import pandas as pd

random.seed(123)
# Define the number of machines to simulate
num_machines = 10

# Define the number of data points to generate for each machine
num_data_points = 100

# Define the range of MTBF values for the machines in hours
min_mtbf = 500
max_mtbf = 2000

# Generate the data for each machine
machine_data = []
machine_mtbf = []

for i in range(num_machines):
    mtbf = random.randint(min_mtbf, max_mtbf)
    machine_mtbf.append(mtbf)
    data = [random.expovariate(1/mtbf) for j in range(num_data_points)]
    machine_data.append(data)

# Calculate the mean MTBF for each machine
mean_mtbf = [sum(data)/len(data) for data in machine_data]

# Determine the index of the machine with the lowest mean MTBF
worst_machine_index = mean_mtbf.index(min(mean_mtbf))

machine_data = pd.DataFrame(machine_data).round()

machine_data.to_csv('data.csv')
# Print the results
print("Machine performance data:")
for i in range(num_machines):
    print(f'Machine {i+1}, MTBF: {round(mean_mtbf[i], 2)} hours')
    
print("Worst performing machine is machine", worst_machine_index+1)
