import numpy as np
import matplotlib.pyplot as plt
import scipy

from ellipsoid_fitting import *


def plot_data(ax,data):
    x = data[:,0:1]
    y = data[:,1:2]
    z = data[:,2:3]

    ax.scatter(x,y,z,s=2)


def least_square_estimate(Y,A):
    # Y = A@Xhat
    Xhat=np.linalg.inv(A.T@A)@A.T@Y
    return Xhat

def sphere_fitting(x,y,z):
    n=x.shape[0]

    # Estimation
    Y=x**2+y**2+z**2
    A=np.hstack((2*x,2*y,2*z,np.ones((n,1))))
    Xhat=least_square_estimate(Y,A)
    x0,y0,z0,temp=Xhat.flatten()
    r0=np.sqrt(temp+x0**2+y0**2+z0**2)


    # Residuals
    res=Y-A@Xhat
    # print("Residuals:")
    # print("mean = ",np.mean(res))
    # print("std = ",np.std(res))

    c=np.array([x0,y0,z0])

    return c,r0

def hard_iron_only_calibration(data):
    x = data[:,0:1]
    y = data[:,1:2]
    z = data[:,2:3]
    n=x.shape[0]

    # Estimation
    c,r0=sphere_fitting(x,y,z)

    x0,y0,z0=c.flatten()

    # Calibration    
    xcal=(x-x0)
    ycal=(y-y0)
    zcal=(z-z0)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    # Plot raw data
    ax.scatter(x,y,z,c="red",s=2)

    # Plot estimation
    phi=np.linspace(0,2*np.pi,100)
    theta=np.linspace(0,np.pi,100)
    phi,theta=np.meshgrid(phi,theta)

    xs=x0+r0*np.sin(theta)*np.cos(phi)
    ys=y0+r0*np.sin(theta)*np.sin(phi)
    zs=z0+r0*np.cos(theta)

    ax.plot_surface(xs,ys,zs,alpha=0.5)

    # Plot calibrated data
    ax.scatter(xcal,ycal,zcal,c="green",s=2)

    ax.set_aspect('equal')
    plt.show()
    

def hard_and_soft_iron_calibration(data):
    
    x = data[:,0:1]
    y = data[:,1:2]
    z = data[:,2:3]

    phat=ellipsoid_fitting(x,y,z)
    Ae,v=extract_ellipsoid_params(phat)
    M=scipy.linalg.sqrtm(Ae)

    return M,v

def print_calibration_parameters(M,v):
    print(f"Soft-iron matrix M =")
    print(M)
    print(f"Hard-iron offset v =")
    print(v)
    print("B_cal = M ( B_m - v )")

def correct_data(data,M,v):
    data=data.T
    data_cor=M@(data-v)
    return data_cor.T

def show_correction(data,data_cor):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    # Plot raw data
    plot_data(ax,data)

    plot_data(ax,data_cor)

    ax.set_aspect('equal')
    plt.show()


def compare_methods(data):

    x = data[:,0:1]
    y = data[:,1:2]
    z = data[:,2:3]

    phat=ellipsoid_fitting(x,y,z)
    Ae,v=extract_ellipsoid_params(phat)
    M=scipy.linalg.sqrtm(Ae)

    c,r0=sphere_fitting(x,y,z)
    x0,y0,z0=c.flatten()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    # Plot raw data
    ax.scatter(x,y,z,c="red",s=2)

    # Plot sphere estimation
    phi=np.linspace(0,2*np.pi,100)
    theta=np.linspace(0,np.pi,100)
    phi,theta=np.meshgrid(phi,theta)

    xs=x0+r0*np.sin(theta)*np.cos(phi)
    ys=y0+r0*np.sin(theta)*np.sin(phi)
    zs=z0+r0*np.cos(theta)

    ax.plot_surface(xs,ys,zs,color="green",alpha=0.5)

    # Plot ellipsoid estimation

    xs=np.sin(theta)*np.cos(phi)
    ys=np.sin(theta)*np.sin(phi)
    zs=np.cos(theta)

    S=np.vstack((xs.ravel(),ys.ravel(),zs.ravel()))
    E=np.linalg.inv(M)@S+v
    
    xe=E[0].reshape(xs.shape)
    ye=E[1].reshape(ys.shape)
    ze=E[2].reshape(zs.shape)

    ax.plot_surface(xe,ye,ze,color="blue",alpha=0.5)

    ax.set_aspect('equal')
    plt.show()
    

if __name__=="__main__":

    mag_data_path="data/mag_data.txt"
    data=np.loadtxt(mag_data_path)

    x = data[:,0:1]
    y = data[:,1:2]
    z = data[:,2:3]
    
    #hard_iron_only_calibration(data)

    # x,y,z=generate_noised_ellipsoid()
    # data=np.hstack((x,y,z))

    M,v=hard_and_soft_iron_calibration(data)
    print_calibration_parameters(M,v)

    data_cor=correct_data(data,M,v)

    show_ellipsoid_fitting(x,y,z,np.linalg.inv(M),v)
    show_correction(data,data_cor)

    

