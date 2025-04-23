import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.linalg

def rot_mat_2D(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],
                     [np.sin(theta),np.cos(theta)]])

def generate_noised_ellipse():
    n=1000
    theta=np.linspace(0,0.5*2*np.pi,n)

    xc=np.cos(theta)
    yc=np.sin(theta)
    C=np.vstack((xc,yc))

    M=rot_mat_2D(np.pi/4)@np.diag([1,3])
    E=M@C

    r=0.2
    x=E[0].reshape((n,1))+r*(np.random.random((n,1))-0.5)
    y=E[1].reshape((n,1))+r*(np.random.random((n,1))-0.5)

    return x,y


def ellipse_fitting(x,y):
    n=x.shape[0]

    # p=(a,b,c,d,e,f)
    D=np.hstack((x**2,y**2,x*y,x,y,np.ones((n,1))))
    S=D.T@D

    # Constraint! 4ab-c**2=1
    u=np.array([[1,0,0,0,0,0]]).T
    v=np.array([[0,1,0,0,0,0]]).T
    w=np.array([[0,0,1,0,0,0]]).T
    M=2*u@v.T+2*v@u.T-w@w.T
    
    eigvals, eigvecs = scipy.linalg.eig(S, M)
    idx = np.flatnonzero((eigvals > 0) * (eigvals!=np.inf))
    phat=eigvecs[:,idx[0]]

    # print("Eigen values: ", eigvals)
    # print("Eigen vectors: ",eigvecs)

    return phat

def extract_ellipse_params(p):

    # Unpack parameters
    a,b,c,d,e,f = p

    # Form matrices
    A = np.array([[a, c/2],
                  [c/2, b]])
    B = np.array([[d],
                  [e]])

    # # Compute center
    center = -0.5 * np.linalg.inv(A) @ B
    x0, y0 = center.flatten()

    # Compute constant term at the center
    fc = (a*x0**2 + b*y0**2 + c*x0*y0 + d*x0 + e*y0 + f)

    # Eigen-decomposition of A for axes lengths and rotation
    eigvals, eigvecs = np.linalg.eigh(A)

    # Sort eigenvalues (larger = semi-major axis)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Semi-axes lengths
    axis_lengths = np.sqrt(-fc / eigvals)
    r1,r2 = axis_lengths

    # Rotation angle (in radians)
    theta= np.arctan2(eigvecs[1, 0], eigvecs[0, 0]) # angle of the minor axis

    return x0, y0, r1, r2, theta

def plot_ellipse_fitting(x,y,phat):
    fig = plt.figure(figsize=(10,10))

    # Plot raw data
    plt.scatter(x,y,c="red",s=2)

    x0, y0, r1, r2, theta = extract_ellipse_params(phat)

    # Parametric ellipse
    n=100
    t=np.linspace(0,2*np.pi,n)
    xc=np.cos(t)
    yc=np.sin(t)
    C=np.vstack((xc,yc))

    M=rot_mat_2D(theta)@np.diag([r1,r2])
    E=M@C
    x_fit=E[0,:]+x0
    y_fit=E[1,:]+y0

    plt.plot(x_fit,y_fit, label="Fitted ellipse", c="green")

    plt.axis("equal")
    plt.show()


if __name__=="__main__":

    x,y=generate_noised_ellipse()
    phat=ellipse_fitting(x,y)
    plot_ellipse_fitting(x,y,phat)
