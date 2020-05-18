import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import csv
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Qt5Agg')



def plot(filename, metric, x, y):
    recs_obs, attack_recs, attack_precision = [], [], []
    my_dict = {}
    
    label = filename[:3]
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reversed(list(reader)):
            
            if row['User'] in my_dict:
                if float(row[metric]) > my_dict[row['User']][0]:
                    my_dict[row['User']] = [float(row[metric]), float(row[x]), float(row[y])]   
            else:
                my_dict[row['User']] = [float(row[metric]), float(row[x]), float(row[y])]

    my_dict = sorted(my_dict.items(), key = lambda kv: kv[1][0], reverse=True)

    for t in my_dict:
        attack_precision.append(t[1][0])
        recs_obs.append(t[1][1])
        attack_recs.append(t[1][2])

    x = recs_obs
    y = attack_recs
    z = attack_precision
    z_label = "Attack "+ metric
    col = np.arange(len(z))

    # 3D Plot
    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
    p3d = ax3D.scatter(x, y, z, s=30, c=col, marker='o')                                                                                

    ax3D.set_title(label)

    # Change the axes labels everytime you change the variables to plot
    ax3D.set_xlabel('Victim RMSE')
    ax3D.set_ylabel('Attacker RMSE')
    ax3D.set_zlabel(z_label)
    print(metric)
    print(y)
    saveas = "Viz/" + label + "_rmse_ermse_" + metric + ".png"
    plt.savefig(saveas, dpi=300)
    plt.show()

if __name__ == "__main__":
    # Model_name, X axis, Y axis, "Metric"
    plot("icf_icf.csv", "Recall", "RMSE", "Eval_RMSE")