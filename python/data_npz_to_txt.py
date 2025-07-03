import numpy as np

data_path = "data/raw_data.npz"

t0=0
tf=-1  # Use -1 to load the entire dataset

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
i_t0 = np.argmin(np.abs(time - t0))

if tf == -1:
    i_tf = N-1
else:
    i_tf = np.argmin(np.abs(time - tf))

t0= time[i_t0]
tf = time[i_tf]

print(f"t0: {t0} s")
print(f"tf: {tf} s")
print("i_t0:", i_t0)
print("i_tf:", i_tf)

mag_data_path="data/mag_data.txt"
with open(mag_data_path,"w") as f:
    for i in range(i_t0, i_tf+1):
        f.write(str(mag_x[i]))
        f.write(" ")
        f.write(str(mag_y[i]))
        f.write(" ")
        f.write(str(mag_z[i]))
        if i!=N-1:
            f.write("\n")
