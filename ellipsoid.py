# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:15:46 2017

@author: Mate
"""

import numpy as np
from math import sin,cos
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,PathPatch
from matplotlib.path import Path


x_axis = np.array([1,0])


def center(E):
    return E[0]

def area(E):
    return abs(np.linalg.det(E[1])*np.pi)

def normalize(x):
    return x / norm(x)

def violated_normal(z,G):
    for p,n in G:
        if n.T.dot(z-p) > 0:
            return n
    else:
        return None

def cut_ellipsoid(E,normal):
    c,A = E
    v = np.array([-normal[1],normal[0]])#iranyvektort csinalok a normalisbol hogy ne kelljen szorakozni   
    v = normalize(np.linalg.inv(A).dot(v)) #a matrix trafot invertalom, igy mar az egyseggombot kell vagnom
    
    n = np.array([v[1],-v[0]]) #normalist csinalok belole
    RT = np.array([[n[0],v[0]],
                   [n[1],v[1]]]) #ez a matrix adja meg hogyan kell majd elgorgatni a dolgokat y-nal vagas utan 
    c1 = np.array([-1/3,0])
    a = 2/3
    b = 2/np.sqrt(3)
    A1 = np.array([[a,0],
                   [0,b]])
    trafo = A.dot(RT)
    c_final = trafo.dot(c1) + c
    A_final = trafo.dot(A1)
    return (c_final,A_final)
    
    
def ellipsoid2d(fd,G,E,epsilon = 0.01,v_epsilon = 0.001):
    z = center(E)
    diff = fd(z)
    while norm(diff) > epsilon and area(E)>v_epsilon:
        rule_normal = violated_normal(z,G)
        cutting_normal = rule_normal if rule_normal is not None else diff
        E = cut_ellipsoid(E,cutting_normal)
        z = center(E)
        diff = fd(z)
    return z

def ellipsoid2d_drawn(fd,G,E,plotter,vert,epsilon = 0.01,v_epsilon = 0.001):
    z = center(E)
    diff = fd(z)
    E_prev=E
    draw(G,E,E_prev,plotter,vert)
    while norm(diff) > epsilon and area(E)>v_epsilon:
        rule_normal = violated_normal(z,G)
        cutting_normal = rule_normal if rule_normal is not None else diff
        E_prev=E        
        E = cut_ellipsoid(E,cutting_normal)
        z = center(E)
        diff = fd(z)
        draw(G,E,E_prev,plotter,vert)
    return z


def get_ell(E,col):
    c,A = E
    a = A[:,0]
    b = A[:,1]
    print(a)
    print(b)
    print(c)
    angle = np.arctan2(b[1],b[0])-np.pi/2
    #angle = np.arctan2(a[1],a[0])
    print(angle/np.pi)
    return Ellipse(xy=c, width=norm(a)*2, height=norm(b)*2, angle=(angle*180/np.pi),alpha=0.3,color=col)

def draw(G,E,E_prev,ax,vert):
    """ax.clear()
    c,A = E
    ell=get_ell(E,(0,0,1))
    cp = Ellipse(xy=c, width=0.1, height=0.1, angle=0,color=(1,0,0))
    ell_prev = get_ell(E_prev,(0,1,0))     
    verts = vert + [vert[0]]
    codes = [Path.MOVETO] + [Path.LINETO for e in range(len(vert)-1)] + [Path.CLOSEPOLY]
    path = Path(verts,codes)
    patch = PathPatch(path, facecolor='orange', lw=2)
    ax.clear()    
    ax.add_patch(patch)
    ax.add_artist(ell_prev)
    ax.add_artist(ell)
    ax.add_artist(cp)
    plt.pause(1)"""
    x,y = get_scatter(E)
    x.append(E[0][0])
    y.append(E[0][1])
    x_prev,y_prev = get_scatter(E_prev)
    verts = vert + [vert[0]]
    codes = [Path.MOVETO] + [Path.LINETO for e in range(len(vert)-1)] + [Path.CLOSEPOLY]
    path = Path(verts,codes)
    patch = PathPatch(path, facecolor='orange', lw=2,alpha=0.5)
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10) 
    ax.add_patch(patch)
    ax.scatter(x,y)
    ax.scatter(x_prev,y_prev)
    plt.pause(0.5)
    
def line(p1,p2):
    v = p2-p1
    n = np.array([v[1],-v[0]])
    return p1,n

def gen_rules(poly):
    n = len(poly)
    return [line(poly[i],poly[(i+1)%n]) for i in range(n)]


def get_scatter(E):
    v = [E[1].dot(np.array([cos(ang),sin(ang)])) for ang in np.linspace(-np.pi,np.pi)]
    x = [e[0]+E[0][0] for e in v]
    y = [e[1]+E[0][1] for e in v]
    return x,y


if __name__ == '__main__':
    G = [ 
          (np.array([1,0]), np.array([1,0])),
          (np.array([-1,0]), np.array([-1,0])),
          (np.array([0,1]), np.array([0,1])),
          (np.array([0,-1]), np.array([0,-1])),
        ]
    vert = [ #CCW felsorolt konvex polygon csucsai, minden csucs pontosan 1x, a szabalyokbol megkaphato lenne
            np.array([1,-1]),
            np.array([1,1]),
            np.array([-1,1]),
            np.array([-1,-1])
            ]
    G = gen_rules(vert) #polygon-> megkotesek
    
    c = np.array([2,1.5])
    a = 8
    b = 12
    R = np.array([[cos(np.pi/4),-sin(np.pi/4)],
                  [sin(np.pi/4), cos(np.pi/4)]])
    A = np.array([[a,0],
                  [0,b]])
    B = R.dot(A)
    E = (c,B)
    fd = lambda x : 2*x.T
    
    plt.ion()
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)    
    
    z = ellipsoid2d_drawn(fd,G,E,ax,vert)
    print(z)