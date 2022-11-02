import matplotlib.pyplot  as plt
import numpy as np

def draw_3d_line(x,y,z):
    fig=plt.figure("1")
    ax = fig.add_subplot(111,projection='3d')

    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    ax.plot(x,y,z,label="curve")
    ax.legend()
    plt.show()

def draw_2d_scatter(x,y,c,m,label,flag):
    fig=plt.figure(str(flag))
    ax = fig.add_subplot(111)
    ax.plot(x, y, "co", marker=m,c=c ,label=label)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.legend()

def draw_3d_scatter(x,y,z,c,m,label,n):
    fig=plt.figure(str(n))
    ax = fig.add_subplot(111,projection='3d')

    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    # ax.plot(x,y,z,label="curve")
    ax.scatter(x, y, z, c=c, marker=m,label=label)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()