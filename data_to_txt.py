import numpy as np

data_path = "data/raw_data.npz"

data = np.load(data_path)

# Extracting all arrays
time    = data["time"]
accel_x = data["accel_x"]
accel_y = data["accel_y"]
accel_z = data["accel_z"]
gyro_x  = data["gyro_x"]
gyro_y  = data["gyro_y"]
gyro_z  = data["gyro_z"]
mag_x   = data["mag_x"]
mag_y   = data["mag_y"]
mag_z   = data["mag_z"]
temp    = data["temp"]

N=time.shape[0]

mag_data_path="data/mag_data.txt"
with open(mag_data_path,"w") as f:
    for i in range(N):
        f.write(str(mag_x[i]))
        f.write(" ")
        f.write(str(mag_y[i]))
        f.write(" ")
        f.write(str(mag_z[i]))
        if i!=N-1:
            f.write("\n")
