import sys
import os
import numpy as np
from magnetometer_calibration import hard_and_soft_iron_calibration, print_calibration_parameters

if __name__=="__main__":

    data_path=sys.argv[1]

    ext=os.path.splitext(data_path)[1]
    if ext==".txt":
        data=np.loadtxt(data_path)
        n,m=data.shape
        if m!=3:
            print("Error: make sure your data are stored in 3 columns for x,y,z")
    elif ext==".npz":
        try:
            data = np.load(data_path)
            x = data["mag_x"]
            y = data["mag_y"]
            z = data["mag_z"]
            n=x.shape[0]
            data=np.hstack((x.reshape(n,1),y.reshape(n,1),z.reshape(n,1)))
        except:
            print("Error reading .npz file.\n" \
            "Check if your magnetometers data are well named: mag_x, mag_y, mag_z")
    elif ext=="":
        print("Error: this program needs your data set path as argument (.txt or .npz)")
    else:
        print("ERROR: your data file extension is incorrect, it must be .txt or .npz")
    
    try:
        M,v=hard_and_soft_iron_calibration(data)
        print_calibration_parameters(M,v)
    except:
        print("Error during calibration")
