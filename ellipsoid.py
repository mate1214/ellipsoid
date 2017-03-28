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
from itertools import chain

class Poly:
    def __init__(self,vs):
        self.verts = vs[:]
        self.lines = [line(p1,p2) for p1,p2 in zip(self.verts[:-1],self.verts[1:])]
        self.lines.append(line(self.verts[-1],self.verts[0]))
        self.cent = self.calccent()
        
    def area(self):
        v=self.verts
        return (sum(a[0]*b[1]-b[0]*a[1] for (a,b) in zip(v[:-1],v[1:])) + v[-1][0]*v[0][1] - v[-1][1]*v[0][0])/2

    def calccent(self):
        v = self.verts
        Cx = (sum((a[0]+b[0])*(a[0]*b[1]-a[1]*b[0]) for (a,b) in zip(v[:-1],v[1:]))
              + (v[-1][0] + v[0][0]) * (v[-1][0]*v[0][1] - v[-1][1]*v[0][0]))
        Cy = (sum((a[1]+b[1])*(a[0]*b[1]-a[1]*b[0]) for (a,b) in zip(v[:-1],v[1:]))
              + (v[-1][1] + v[0][1]) * (v[-1][0]*v[0][1] - v[-1][1]*v[0][0]))
        return np.array([Cx,Cy]) / (6*self.area())

    def cut(self,normal):
        c = self.cent
        verts = self.verts
        negside=[]
        for p1,p2 in zip(verts,chain(verts[1:],[verts[0]])):
            d1,d2 = (p1-c).dot(normal), (p2-c).dot(normal)
            t = np.abs(d1)/(np.abs(d1)+np.abs(d2))
            if d1<=0:
                negside.append(p1)
            if d1*d2<0:
                negside.append((1-t)*p1+t*p2)
        self.verts = negside
        self.cent = self.calccent()   
            


def get_angle(l):
    n = l[1]
    return np.arctan2(n[1],n[0])        

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


    

def polygon_algorithm(fd,vertices,epsilon = 0.01,v_epsilon = 0.001):
    P = Poly(vertices)
    diff = fd(P.cent)
    while norm(diff) > epsilon and P.area()>v_epsilon:   
        P.cut(diff)
        diff = fd(P.cent)
    return P.cent

def poly_draw(pol,vert,ax):
    verts = vert + [vert[0]]
    codes = [Path.MOVETO] + [Path.LINETO for e in range(len(vert)-1)] + [Path.CLOSEPOLY]
    patch = PathPatch(Path(verts,codes), facecolor='orange', lw=2,alpha=0.5)
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10) 
    ax.add_patch(patch)
    verts = pol.verts + [pol.verts[0]]
    codes = [Path.MOVETO] + [Path.LINETO for e in range(len(verts)-2)] + [Path.CLOSEPOLY]
    patch = PathPatch(Path(verts,codes), facecolor='green', lw=2,alpha=0.5)
    ax.add_patch(patch)
    plt.pause(1.0)
    
def polygon_algorithm_drawn(fd,vertices,ax,epsilon = 0.01,v_epsilon = 0.001):
    P = Poly(vertices)
    diff = fd(P.cent)
    areas=[P.area()]
    while norm(diff) > epsilon and P.area()>v_epsilon:   
        P.cut(diff)
        diff = fd(P.cent)
        poly_draw(P,vertices,ax)
        areas.append(P.area())
    return P.cent,areas
    
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
    draw(G,E,E_prev,None,plotter,vert)
    areas = [area(E)]
    while norm(diff) > epsilon and area(E)>v_epsilon:
        rule_normal = violated_normal(z,G)
        cutting_normal = rule_normal if rule_normal is not None else diff
        E_prev=E        
        E = cut_ellipsoid(E,cutting_normal)
        z = center(E)
        diff = fd(z)
        draw(G,E,E_prev,cutting_normal,plotter,vert)
        areas.append(area(E))
    return z,areas


def get_ell(E,col):
    c,A = E
    w,v = np.linalg.eig(A)
    if w[0]<w[1]:
        a,b=w[0],w[1]
        angle = np.arctan2(np.absolute(v[1,0]),np.absolute(v[0,0]))
    else:
        a,b=w[1],w[0]
        angle = np.arctan2(np.absolute(v[1,1]),np.absolute(v[0,1]))
    return Ellipse(xy=c, width=a*2, height=b*2, angle=(angle*180/np.pi),alpha=0.3,color=col)

def draw(G,E,E_prev,cutting_normal,ax,vert):
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
    plt.pause(2)"""
    x,y = get_scatter(E)
    x.append(E[0][0])
    y.append(E[0][1])
    x_prev,y_prev = get_scatter(E_prev)
    verts = vert + [vert[0]]
    codes = [Path.MOVETO] + [Path.LINETO for e in range(len(vert)-1)] + [Path.CLOSEPOLY]
    patch = PathPatch(Path(verts,codes), facecolor='orange', lw=2,alpha=0.5)
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10) 
    ax.add_patch(patch)
    if cutting_normal is not None:
        ax.add_patch(PathPatch(get_line(E_prev[0],cutting_normal), facecolor='blue', lw=5,alpha=0.5))
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

def get_line(p,n):
    v = np.array([-n[1],n[0]])
    verts=[p+6*v, p-6*v]
    codes = [Path.MOVETO, Path.LINETO]
    return Path(verts,codes)


def run_poly():
    vert = [ #CCW felsorolt konvex polygon csucsai, minden csucs pontosan 1x, a szabalyokbol megkaphato lenne
            np.array([1,-1]),
            np.array([1,1]),
            np.array([-1,1]),
            np.array([-1,-1])
            ]    
    fd = lambda x : 2*(x.T - [0.5,0.5])
    plt.ion()
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)    
    
    z,areas = polygon_algorithm_drawn(fd,vert,ax)
    print(z)
    print(len(areas)-1)   

def run_ellipse():
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
    #fd = lambda x : 2*x.T
    fd = lambda x : 2*(x.T - [0.5,0.5])
    plt.ion()
    plt.axis('equal')
    ax = plt.gca()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)    
    
    z,areas = ellipsoid2d_drawn(fd,G,E,ax,vert)
    print(z)
    print(len(areas)-1)
    
if __name__ == '__main__':
    run_poly()
    #run_ellipse()
