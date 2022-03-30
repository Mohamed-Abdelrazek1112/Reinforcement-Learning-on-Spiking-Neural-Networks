import gym
from gym import wrappers
import matplotlib.pyplot as plt
from IPython import display
import random
import cv2
import pickle
import pyNN.spiNNaker as p
p.setup(1)
import math

def inputFrameProcessor(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
def inputToSpikeSourceArray(frame):
    grayframe = inputFrameProcessor(frame) 
    return [m for line in grayframe for m in line]

def createInputLayer(image_size):
    inputLayer = p.Population(image_size,p.SpikeSourcePoisson(rate=[0]*image_size))
    inputLayer.record(["spikes"])
    return inputLayer

def createHiddenLayer(number_of_actions,npa):
    excitationLayer = []
    inhibitoryLayer = []
    for i in range(number_of_actions):
        e = p.Population(npa,p.IF_curr_exp())
        i = p.Population(npa,p.IF_curr_exp())
        i.record(["spikes"])
        e.record(["spikes"])
        excitationLayer.append(e)
        inhibitoryLayer.append(i)
    return excitationLayer,inhibitoryLayer

def selectOutput(records):#REPLACE WITH SOFTMAX
    choice = 0
    best = 0
    for i in range(len(records)):
        if len(records[i])>best:
            best = len(records[i])
            choice = i
    return choice

timing_rule = p.SpikePairRule(tau_plus=0.1, tau_minus=0.1, A_plus=0.1, A_minus=0.1)
weight_rule_pos = p.AdditiveWeightDependence(w_max=50.0, w_min=1.0)
stdp_model_excitatory = p.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule_pos, weight=25.0)

def inputExcitationProjections(inp,exc,w,prob):
    p.Projection(inp,exc,p.FixedProbabilityConnector(prob), synapse_type=p.StaticSynapse(weight=w))
def excitationinhibitionProjections(exc,inh,syn):
    p.Projection(exc,inh,p.AllToAllConnector(), synapse_type=syn)
def inhibitionexcitationProjections(exc,inh,w):
    p.Projection(exc,inh,p.AllToAllConnector(), synapse_type=p.StaticSynapse(weight=w))
def trainingSignal(actions):
    trainLayer = p.Population(actions,p.IF_curr_exp(),p.SpikeSourcePoisson(rate=[0]*actions))
    trainLayer.record(["spikes"])
    return trainLayer
def trainingProjections(train,excL,inhL,w):
    for i in range(len(train)):
        p.Projection(train[i],excL[i],p.AllToAllConnector(), synapse_type=p.StaticSynapse(weight=w))
        p.Projection(train[i],inhL[i],p.AllToAllConnector(), synapse_type=p.StaticSynapse(weight=w))

with open("pickeledDataPong","rb") as file:
    TRAINING_DATA = pickle.load(file)

inputLayer = createInputLayer(33600)
excitationLayer,inhibitionLayer = createHiddenLayer(6,10)
a = [inputExcitationProjections(inputLayer,exc,1,0.8) for exc in excitationLayer]


for data in TRAINING_DATA:
    image,action = data
    inputLayer.set(rate=image)
    #trainLayer[action].set(rate=100)
    break