import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import csv
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Qt5Agg')

recs_obs, purchase_history, attack_precision = [], [], []

my_dict = {}

with open('Movies_results_icf.csv', 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        
        if row['User'] in my_dict:
            if float(row['Precision']) > my_dict[row['User']][0]:
                my_dict[row['User']] = [float(row['Precision']), float(row['Num of Recs(K_O)']), float(row['Original Items'])]   
        else:
            my_dict[row['User']] = [float(row['Precision']), float(row['Num of Recs(K_O)']), float(row['Original Items'])]

my_dict = sorted(my_dict.items(), key = lambda kv: kv[1][0], reverse=True)
print(my_dict)

for t in my_dict:
    attack_precision.append(t[1][0])
    recs_obs.append(t[1][1])
    purchase_history.append(t[1][2])

x = purchase_history
y = recs_obs
z = attack_precision
col = np.arange(218)

# 3D Plot
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
p3d = ax3D.scatter(x, y, z, s=30, c=col, marker='o')                                                                                

ax3D.set_xlabel('Purchase History')
ax3D.set_ylabel('Recommendations Observed')
ax3D.set_zlabel('Attack Precision')

plt.show()