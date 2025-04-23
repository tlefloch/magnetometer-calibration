import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.spatial.transform import Rotation

def euler_to_rotmat(phi,theta,psi):
    Rx=np.array([[1,0,0],
                 [0,np.cos(phi),-np.sin(phi)],
                 [0,np.sin(phi),np.cos(phi)]])
    Ry=np.array([[np.cos(theta),0,np.sin(theta)],
                 [0,1,0],
                 [-np.sin(theta),0,np.cos(theta)]])
    Rz=np.array([[np.cos(psi),-np.sin(psi),0],
                 [np.sin(psi),np.cos(psi),0],
                 [0,0,1]])
    return Rz@Ry@Rx


def generate_noised_ellipsoid():

    n=100

    phi=np.linspace(0,2*np.pi,n)
    theta=np.linspace(0,np.pi,n)
    phi,theta=np.meshgrid(phi,theta)

    xs=np.sin(theta)*np.cos(phi)
    ys=np.sin(theta)*np.sin(phi)
    zs=np.cos(theta)
    S=np.vstack((xs.ravel(),ys.ravel(),zs.ravel()))

    # = Rotation.from_euler("ZYX",[90, 45, 30], degrees=True).as_matrix()
    phi,theta,psi=np.pi*(2*np.random.random((3,))-1)
    R=euler_to_rotmat(phi,theta,psi)

    a,b,c=4*np.random.random((3,))+1
    M=R@np.diag([a,b,c])
    E=M@S

    x=E[0].reshape(-1,1)
    y=E[1].reshape(-1,1)
    z=E[2].reshape(-1,1)
    
    i=np.random.random((x.shape[0],))<=0.5
    x=x[i]
    y=y[i]
    z=z[i]

    r=0.2
    x=x+r*(np.random.random(x.shape)-0.5)
    y=y+r*(np.random.random(x.shape)-0.5)
    z=z+r*(np.random.random(x.shape)-0.5)

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(projection='3d')

    # # Plot raw data
    # ax.scatter(x,y,z,c="red",s=2)

    # ax.set_aspect('equal')
    # plt.show()

    return x,y,z

def ellipsoid_fitting(x,y,z):
    n=x.shape[0]
    D=np.hstack((x**2,y**2,z**2,x*y,y*z,z*x,x,y,z,np.ones((n,1))))
    S=D.T@D

    not_ellipsoid=True
    cnt=0
    while not_ellipsoid:
        cnt+=1

        alpha=2*np.random.random()-1
        beta=2*np.random.random()-1
        gamma=2*np.random.random()-1

        # Constraint: plane intersection => ellipse
        u=np.array([[1,0,alpha**2,0,0,alpha,0,0,0,0]]).T
        v=np.array([[0,1,beta**2,0,beta,0,0,0,0,0]]).T
        w=np.array([[0,0,2*alpha*beta,1,alpha,beta,0,0,0,0]]).T
        M=2*u@v.T+2*v@u.T-w@w.T

        eigvals, eigvecs = scipy.linalg.eig(S, M)
        idx = np.flatnonzero((eigvals > 0) * (eigvals!=np.inf))

        # print("Eigen values: ", eigvals)
        # print("Eigen vectors: ",eigvecs)
        # print(idx)

        phat=eigvecs[:,idx[0]]



        a,b,c,d,e,f,g,h,i,j=phat.flatten()

        Q=np.array([[a,d/2,f/2],
                    [d/2,b,e/2],
                    [f/2,e/2,c]])
        eigvals, eigvecs = scipy.linalg.eigh(Q)

        cond1=eigvals[0]*np.prod(eigvals)>0

        P=np.array([[a,d/2,f/2,g/2],
                    [d/2,b,e/2,h/2],
                    [f/2,e/2,c,i/2],
                    [g/2,h/2,i/2,j]])
        
        cond2=scipy.linalg.det(P)!=0

        if cond1 and cond2:
            not_ellipsoid=False
    print("Number of random plans tested: ",cnt)

    # To insure A0 is positive definite and k is positive
    # in the expression (x-v).T @ A0 @ (x-v) = k
    # if np.all(eigvals<0):
    #     phat=-phat

    return phat

def extract_ellipsoid_params(p):
    a,b,c,d,e,f,g,h,i,j=p.flatten()

    A=np.array([[a,d/2,f/2],
                [d/2,b,e/2],
                [f/2,e/2,c]])
    b=np.array([[g,h,i]]).T

    A0=A
    v=-1/2*np.linalg.inv(A0)@b
    k=v.T@A0@v-j

    Ae=A0/k

    return Ae,v

def plot_ellipsoid(ax,M,t):

    # Plot estimation
    phi=np.linspace(0,2*np.pi,100)
    theta=np.linspace(0,np.pi,100)
    phi,theta=np.meshgrid(phi,theta)

    xs=np.sin(theta)*np.cos(phi)
    ys=np.sin(theta)*np.sin(phi)
    zs=np.cos(theta)

    S=np.vstack((xs.ravel(),ys.ravel(),zs.ravel()))
    E=M@S+t
    
    xe=E[0].reshape(xs.shape)
    ye=E[1].reshape(ys.shape)
    ze=E[2].reshape(zs.shape)

    ax.plot_surface(xe,ye,ze,color="blue",alpha=0.5)



def show_ellipsoid_fitting(x,y,z,M,t):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    # Plot raw data
    ax.scatter(x,y,z,c="red",s=2)

    # Plot ellipsoid
    plot_ellipsoid(ax,M,t)

    ax.set_aspect('equal')
    plt.show()

if __name__=="__main__":

    x,y,z=generate_noised_ellipsoid()
    phat=ellipsoid_fitting(x,y,z)
    Ae,v=extract_ellipsoid_params(phat)
    M=np.linalg.inv(scipy.linalg.sqrtm(Ae))
    show_ellipsoid_fitting(x,y,z,M,v)
