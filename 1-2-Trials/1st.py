import pyNN.spiNNaker as p
p.setup(timestep=20)

def inputFrameProcessor(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
def inputToSpikeRateArray(frame):
    grayframe = inputFrameProcessor(frame) 
    grayframe= cv2.resize(grayframe,[84,84])
    return [(m+1) for line in grayframe for m in line]
def selectOutput(records,previous_spikes):
    choice = 0
    best = 0
    j=-1
    new_previous_spikes=[]
    if previous_spikes==[]:
        previous_spikes=[0]*len(records)
    for rec in records:
        j+=1
        #print("action",j,"spikes",len(rec)-previous_spikes[j])
        new_previous_spikes.append(len(rec))
    for i in range(len(records)):
        if (len(records[i])-previous_spikes[i])>best:
            best = len(records[i])-previous_spikes[i]
            choice = i
    return choice,new_previous_spikes
FRAME=0
TRAIN=0
ACTIONS = 4

##### INPUT LAYER #####
inputLayer = p.Population(84*84,p.SpikeSourcePoisson(rate=inputToSpikeRateArray(FRAME)))
inputLayer.record(["spikes"])
#######################


#### OUTPUT LAYER ####
outputLayer=[p.Population(int(100),p.IF_curr_exp()) for action in range(ACTIONS)]
s = [m.record(["spikes"]) for m in pop]
######################

##### TRAINING REWARD INPUT ######
if TRAIN:
    pre_rewardLayer = [p.Population(100) for _ in range(ACTIONS)]
    post_rewardLayer = [p.Population(100) for _ in range(ACTIONS)]
###################################

#####   STDP    #####
timing_rule = p.SpikePairRule(tau_plus=0.1, tau_minus=0.1, A_plus=0.1, A_minus=0.1)
weight_rule = p.AdditiveWeightDependence(w_max=10.0, w_min=0.01)
stdp_model = p.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule, weight=5)
#####################

#### OUTPUT LAYER ####
pop=[p.Population(int(100),p.IF_curr_exp()) for action in range(ACTIONS)]
s = [m.record(["spikes"]) for m in pop]
######################

#### Projections ####
projections01= [p.Projection(inputLayer,m,p.AllToAllConnector(), synapse_type=stdp_model) for m in pop]
rewardproj=[]
punishmentproj = []
for i in range(env.action_space.n):
    rewardproj.append(p.Projection(rewardLayer[i],pop[i],p.AllToAllConnector(),synapse_type=p.StaticSynapse(weight=0.11)))
    punishmentproj.append(p.Projection(punishmentLayer[i],pop[i],p.AllToAllConnector(), synapse_type=p.StaticSynapse(weight=-50)))






'''
Linear learners are bad at breakout and pong; According to Minh et. al[REF Minh] the results show that linear learners are 
poor at the tasks of breakout and pong achieving performances well below that of humans.
While SNNs are not linear learners this gives insight that classifications cannot play games well.
'''