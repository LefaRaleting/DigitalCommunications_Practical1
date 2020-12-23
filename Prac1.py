# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:09:43 2020

@author: Boris
"""
import random
import math
from datetime import datetime
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm
import mpl_toolkits.mplot3d 
from operator import add
import copy


def Whichman():
    # creating the random values based on the above seed
    x = random.randint(1, 30000)
    y = random.randint(1, 30000)
    z = random.randint(1, 30000)

    # first generation
    x = 171 * (x % 177) - 2 * (x / 177)

    if x < 0:
        x = x + 30629
        # first generation
    y = 172 * (y % 176) - 2 * (y / 176)
    if y < 0:
        y = y + 30307
    # first Generation
    z = 170 * (z % 178) - 2 * (z / 178)
    if z < 0:
        z = z + 30323
    temp = x / 30269 + y / 30307 + z / 30323

    return temp - math.trunc(temp)


def Whichman_Random_Generator(size):
    values = []
    # creating a random seed based on the current date and time now
    random.seed(datetime.now())
    for i in range(0, size):
        values.append(Whichman())
    return values


# plotting
# size of the random values
size = 98
randomValues = Whichman_Random_Generator(size)
laterValues = copy.deepcopy(randomValues)
mu = st.mean(randomValues)
sigma = st.stdev(randomValues)

x = np.linspace(-1, 1, size)
randomValues.sort()

plt.plot(randomValues, norm(mu, sigma).pdf(randomValues))
plt.ylabel('Probability Density')
plt.xlabel('Randomly Generated Numbers')
plt.show()

print("Sigma:", sigma)
print("Mu:", mu)
print("Size: ", size)


# _____________________________________________________________________________
#                       Task2
# ____________________________________________________________________________

def Gaussian(Seed):
    length = Seed
    loop = 0
    
    GaussX = []
    GaussY = []
    
     
    while loop < length: 
        v = [random.random(), random.random()]
    
        v[0] = 2*v[0]-1
        v[1] = 2*v[1]-1
        while (v[0]**2 + v[1]**2 > 1) or (v[0]**2+ v[1]**2 == 0):
            v[0] = random.random()
            v[1] = random.random()
            
            v[0] = 2*v[0]-1
            v[1] = 2*v[1]-1
        X = v[0]*(-2*np.log(v[0]**2+v[1]**2)/(v[0]**2+v[1]**2))**0.5
        Y = v[1]*(-2*np.log(v[0]**2+v[1]**2)/(v[0]**2+v[1]**2))**0.5
        GaussX.append(X)
        GaussY.append(Y)
        loop = loop + 1
  
    return GaussX, GaussY

#plotting 
def Plot2D(X = [],Y=[],density=100):
    A = X+Y
    plt.hist(A, density)

    
# 3D map plotting 
# source: ArtifexR, https://stackoverflow.com/questions/8437788/how-to-correctly-generate-a-3d-histogram-using-numpy-or-matplotlib-built-in-func    
def Plot3D(GaussX = [], GaussY = []):
    xAmplitudes = GaussX
    yAmplitudes = GaussY
    
    x = np.array(xAmplitudes)   #turn x,y data into numpy arrays
    y = np.array(yAmplitudes)
    
    fig = plt.figure()          #create a canvas, tell matplotlib it's 3d
    ax = fig.add_subplot(111, projection='3d')
    
    #make histogram stuff - set bins - I choose 20x20 because I have a lot of data
    hist, xedges, yedges = np.histogram2d(x, y, bins=(20,20))
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
    
    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)
    
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()
    
    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.title("Gaussian distribution about 0")
    plt.xlabel("X Value")
    plt.ylabel("Y Value")
    plt.savefig("Gaussian distribution about 0")
    plt.show()


# _____________________________________________________________________________
#                               Bit generator
# _____________________________________________________________________________
# random bits generator
def bits_gen(values):
    data = []
    # values= Whichman_Random_Generator(number)
    for i in values:
        if i < 0.5:
            data.append(0)
        else:
            data.append(1)

    return data


# ________________________________________________________________________________
#               mapping of bits to symbol using constellation maps
# _________________________________________________________________________________
def BPSK(data):
    bpsk = []
    for k in bits:
        if k == 1:
            bpsk.append(1)
        else:
            bpsk.append(-1)
    return bpsk


def fourQAM(data):
    FQAM = []
    M = 2
    subList = [bits[n:n + M] for n in range(0, len(bits), M)]
    for k in subList:
        if k == [0, 0]:
            FQAM.append(complex(1 / np.sqrt(2), 1 / np.sqrt(2)))
        elif k == [0, 1]:
            FQAM.append(complex(-1 / np.sqrt(2), 1 / np.sqrt(2)))
        elif k == [1, 1]:
            FQAM.append(complex(-1 / np.sqrt(2), -1 / np.sqrt(2)))
        # elif(k==[1,0]):
        elif k == [1, 0]:
            FQAM.append(complex(1 / np.sqrt(2), -1 / np.sqrt(2)))

    return FQAM


def eight_PSK(data):
    EPSK = []
    M = 3
    subList = [bits[n:n + M] for n in range(0, len(bits), M)]
    for k in subList:
        if k == [0, 0, 0]:
            EPSK.append(complex(1 / np.sqrt(2), 0))
        elif k == [0, 0, 1]:
            EPSK.append(complex(1 / 2, 1 / 2))
        elif k == [0, 1, 1]:
            EPSK.append(complex(0, 1 / np.sqrt(2)))
        elif k == [0, 1, 0]:
            EPSK.append(complex(-1 / 2, 1 / 2))
        elif k == [1, 1, 0]:
            EPSK.append(complex(-1 / np.sqrt(2), 0))
        elif k == [1, 1, 1]:
            EPSK.append(complex(-1 / 2, -1 / 2))
        elif k == [1, 0, 1]:
            EPSK.append(complex(0, -1 / np.sqrt(2)))
        elif k == [1, 0, 0]:
            EPSK.append(complex(1 / 2, -1 / 2))
    return EPSK


def sixteenQAM(data):
    sixtQAM = []
    M = 4
    subList = [bits[n:n + M] for n in range(0, len(bits), M)]
    for k in subList:
        if k == [0, 0, 0, 0]:
            sixtQAM.append(complex(-3, -3))
        elif k == [0, 0, 0, 1]:
            sixtQAM.append(complex(-3, -1))
        elif k == [0, 0, 1, 1]:
            sixtQAM.append(complex(-3, 1))
        elif k == [0, 0, 1, 0]:
            sixtQAM.append(complex(-3, 3))
        elif k == [0, 1, 1, 0]:
            sixtQAM.append(complex(-1, 3))
        elif k == [0, 1, 1, 1]:
            sixtQAM.append(complex(-1, 1))
        elif k == [0, 1, 0, 1]:
            sixtQAM.append(complex(-1 - 1))
        elif k == [0, 1, 0, 0]:
            sixtQAM.append(complex(-1, -3))
        elif k == [1, 1, 0, 0]:
            sixtQAM.append(complex(1, -3))
        elif k == [1, 1, 0, 1]:
            sixtQAM.append(complex(1, -1))
        elif k == [1, 1, 1, 1]:
            sixtQAM.append(complex(1, 1))
        elif k == [1, 1, 1, 0]:
            sixtQAM.append(complex(1, 3))
        elif k == [1, 0, 1, 0]:
            sixtQAM.append(complex(3, 3))
        elif k == [1, 0, 1, 1]:
            sixtQAM.append(complex(3, 1))
        elif k == [1, 0, 0, 1]:
            sixtQAM.append(complex(3, -1))
        elif k == [1, 0, 0, 0]:
            sixtQAM.append(complex(3, -3))
    return sixtQAM


# ______________________________________________________________________________
#                           Noise addition
# ______________________________________________________________________________

def Add_noise(transmitted, Gnoise, M, SNR):
    gama = 1 / np.sqrt(math.pow(10, (SNR / 10)) * 2 * math.log2(M))
    # print(gama)
    new = [i * gama for i in Gnoise]
    R = list(map(add, transmitted, new))
    return R


# _____________________________________________________________________________
#                           Detection
# ______________________________________________________________________________

def BPSKDetection(comp):
    points = [-1, 1]
    Bpoints = [[0], [1]]
    recieved = -1
    minDistance = 99
    decoded = []
    Bdecoded = []
    for y in comp:
        for x in range(len(points)):
            distance = (y - points[x]) ** 2
            if distance <= minDistance:
                minDistance = distance
                recieved = x
        decoded.append(points[recieved])
        Bdecoded.append(Bpoints[recieved])
    return decoded, Bdecoded  # recieved


def QAM4Detection(comp):
    points = [(1 + 1j) / np.sqrt(2), (-1 - 1j) / np.sqrt(2),
              (1 - 1j) / np.sqrt(2), (-1 + 1j) / np.sqrt(2)]
    Bpoints = [[0, 0], [1, 1], [1, 0], [0, 1]]
    recieved = -1
    minDistance = 99
    decoded = []
    Bdecoded = []
    for y in comp:
        for x in range(len(points)):
            distance = (points[x] - y) ** 2
            if np.abs(distance) <= np.abs(minDistance):
                minDistance = distance
                recieved = x
        decoded.append(points[recieved])
        Bdecoded.append(Bpoints[recieved])
    return decoded, Bdecoded


def PSK8Detection(comp):
    # points = [(-1 - 1j) / np.sqrt(2), -1, 1j,(-1 + 1j) / np.sqrt(2),  -1j, (1 - 1j) / np.sqrt(2), (1 + 1j) /
    # np.sqrt(2), 1]
    points = [complex(1 / np.sqrt(2), 0), complex(1 / 2, 1 / 2),
              complex(0, 1 / np.sqrt(2)), complex(-1 / 2, 1 / 2),
              complex(-1 / np.sqrt(2), 0), complex(-1 / 2, -1 / 2),
              complex(0, -1 / np.sqrt(2)), complex(1 / 2, -1 / 2)]
    Bpoints = [[0, 0, 0], [0, 0, 1],
               [0, 1, 1], [0, 1, 0],
               [1, 1, 0], [1, 1, 1],
               [1, 0, 1], [1, 0, 0]]
    recieved = -1
    minDistance = 99
    decoded = []
    Bdecoded = []
    for y in comp:
        for x in range(len(points)):
            distance = (points[x] - y) ** 2
            if np.abs(distance) <= np.abs(minDistance):
                minDistance = distance
                recieved = x
        decoded.append(points[recieved])
        Bdecoded.append(Bpoints[recieved])
    return decoded, Bdecoded


def QAM16Detection(comp):
    # points = [-1 + 1j, -1 + 1j / 3, -1 - 1j, -1 - 1j / 3,
    #          -1 / 3 + 1j, (-1 + 1j) / 3, -1 / 3 - 1j, (-1 + 1j) / 3,
    #          1 + 1j, 1 + 1j / 3, 1 - 1j, 1 - 1j / 3,
    #          1 / 3 + 1j, (1 + 1j) / 3, 1 / 3 - 1j, (1 - 1j) / 3]
    points = [complex(-3, -3), complex(-3, -1), complex(-3, 1), complex(-3, 3),
              complex(-1, 3), complex(-1, 1), complex(-1, -1), complex(-1, -3),
              complex(1, -3), complex(1, -1), complex(1, 1), complex(1, 3),
              complex(3, 3), complex(3, 1), complex(3, -1), complex(3, -3)]
    Bpoints = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0],
               [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 0, 1], [0, 1, 0, 0],
               [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0],
               [1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]]
    recieved = -1
    minDistance = 99
    decoded = []
    Bdecoded = []
    for y in comp:
        for x in range(len(points)):
            distance = (points[x] - y) ** 2
            if np.abs(distance) <= np.abs(minDistance):
                minDistance = distance
                recieved = x
        decoded.append(points[recieved])
        Bdecoded.append(Bpoints[recieved])
    return decoded, Bdecoded


# __________________________________________________________________________________
#
# ________________________________________________________________________________


# ______________________________________________________________________________
#                           Transmision and detection
# ____________________________________________________________________________
def transmission(n, bits):
    if n == 1:
        return BPSK(bits)
    elif n == 2:
        return fourQAM(bits)
    elif n == 3:
        return eight_PSK(bits)
    elif n == 4:
        return sixteenQAM(bits)


def detection(n, bits):
    if n == 1:
        return BPSKDetection(bits)
    elif n == 2:
        return QAM4Detection(bits)
    elif n == 3:
        return PSK8Detection(bits)
    elif n == 4:
        return QAM16Detection(bits)


def bit_errors(sent, recieved):
    error = 0
    for k in range(len(recieved)):
        if sent[k] != recieved[k]:
            error += 1
    BER = error / len(recieved)

    return BER


def SYM_error(sent, recieved):
    error = 0
    for k in range(len(recieved)):
        if sent[k] != recieved[k]:
            error += 1
    SER = error / len(recieved)
    return SER


# _________________________________________________________________________________
#                                   Task3
# _________________________________________________________________________________
# SNR= -4
# bpsk= 2 , 4QAM=4 , 8psk = 8, 16QAM =16
M = 4
bits = bits_gen(laterValues)  # The size is done at the top   of the code
# Select mapping constellation 1=BPSK , 2=4QAM,3= 8PSK, 4=16QAM
Mode = 2
sent = transmission(Mode, bits)


# Recieved=Add_noise(sent,Ax,M,SNR)
# Detected,Dbits=detection(Mode,Recieved)
# Dbits = [item for sublist in Dbits for item in sublist]
# BER=bit_errors(bits,Dbits)
# SER= SYM_error(sent,Detected)


def SER_BER(Mode, M, bits, SNR):
    BER = []
    SER = []
    for i in SNR:
        Recieved = Add_noise(sent, Ax, M, i)
        Detected, Dbits = detection(Mode, Recieved)
        Dbits = [item for sublist in Dbits for item in sublist]
        BER.append(bit_errors(bits, Dbits))
        SER.append(SYM_error(sent, Detected))
        # print (i)
    return BER, SER


SNR = np.linspace(-4, 12, 50)
# print(list(SNR))
# BER,SER= SER_BER(Mode,M,bits,SNR)
# print(BER)
