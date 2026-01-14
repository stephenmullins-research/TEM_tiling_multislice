

"""
Plotting time tests for Figure 8
"""


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# Define the folder path (adjust path as needed)
folder_path = 'Time_Test'

# List all pickle files in the folder
pickle_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

# Read and load the pickle files as numpy arrays
pickle_data = {}
for pickle_file in pickle_files:
    file_path = os.path.join(folder_path, pickle_file)
    with open(file_path, 'rb') as file:
        pickle_data[pickle_file] = pickle.load(file)

image_list = np.arange(1024,8193,1024)


#Serial or Parallel Computation
fig = plt.figure(figsize=(12,6))
fig.add_subplot(1,2, 1)
plt.plot(image_list,pickle_data['Sequential_Conventional.pkl']/60,'bo-',label='Conventional')
plt.plot(image_list,pickle_data['Sequential_2x2.pkl']/60,'gs--',label='2x2 Tiling')
plt.plot(image_list,pickle_data['Sequential_4x4.pkl']/60,'k^:',label='4x4 Tiling')
plt.title('Serial Processing',fontsize=16)
plt.xlabel('Image Size (pixel)',fontsize=14)
plt.ylabel('Time (min)',fontsize=14)
plt.xticks(image_list[::1],fontsize=14)
plt.yticks(fontsize=14)
plt.legend(prop={'size':14})


fig.add_subplot(1,2, 2)
plt.plot(image_list,pickle_data['Sequential_Conventional.pkl']/60,'bo-',label='Conventional')
plt.plot(image_list,pickle_data['Parallel_2x2.pkl']/60,'gs--',label='2x2 Tiling')
plt.plot(image_list,pickle_data['Parallel_4x4.pkl']/60,'k^:',label='4x4 Tiling')
plt.title('Parallel Processing',fontsize=16)
plt.xlabel('Image Size (pixel)',fontsize=14)
plt.ylabel('Time (min)',fontsize=14)
plt.xticks(image_list[::1],fontsize=14)
plt.yticks(fontsize=14)
plt.legend(prop={'size':14})

plt.tight_layout()
plt.show()
plt.savefig('time_scaling_test_cpu_100_slices.png',dpi=300)


