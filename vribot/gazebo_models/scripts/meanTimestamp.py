import numpy as np

filepath="../experiments/lab_video15_3marker_no_chair/stamped_traj_estimate.txt"
# filepath="../experiments/lab_video15_3marker_no_chair/stamped_groundtruth.txt"

total = []
with open(filepath, "r") as fp:
    line = fp.readline()
    line = fp.readline()
    cnt = 1
    val = line.split(" ")[0]
    val = float(val)
    val_last = val
    # total.append(val)
    while line:
        val = line.split(" ")[0]
        val = float(val)
        new = val-val_last
        val_last = val
        total.append(new)
        line = fp.readline()
        cnt+=1

total = np.array(total)

print(np.mean(total))
print(np.std(total))


# for i in range(0,1001):
#     print(i,i)
#     # print(i,i)
