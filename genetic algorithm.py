# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 21:47:34 2025

@author: Student
"""



#GENETIC ALGORITHM

#PHIxNUE SELECTION


import numpy as np
import PSII_params as PSII
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.colors as colors
from scipy.stats import norm 
import statistics  
import random
import sys
import heapq
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#bring in antenna function
from Lattice_antenna import Antenna_branched_overlap
from Lattice_antenna import gauss_abs
import csv
plt.rcParams.update(plt.rcParamsDefault)
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
font = {'fontname': 'serif'}
plt.figure(dpi=1200)




'''         CONTROLS        '''


spec='2800K'
vers=2         # how many runs
start=6
fin=10

T=100           # TIME

no_subunits=2
N_b=1

start_pop=50
cutoff=10

mutation_rate=0.2



#LOADING wavelengths/flux

file=open('SmoothedSpectrum3000_PHOENIX_2800K_ga.txt','r')

temp='{}.b{}.s{}.v{}'.format(spec,N_b,no_subunits,vers)
            
            



#print(file.read())
l,Ip_y=[],[]
for line in file:
    line=line.rstrip()
    elements=line.split()
    l.append(float(elements[0]))
    Ip_y.append(float(elements[1]))
l=np.array(l)
Ip_y=np.array(Ip_y)

file.close()



#Z
#P  A   R   A   M   E   T   E   R   S
#RC parameters, constant
N_RC=1
sig_RC=1.6E-21 #optical cross section
lp_RC=680.0
w_RC=10.0 #8.9-9.1
RC_params=(N_RC,sig_RC,lp_RC,w_RC)
#Rate constant k parameters, constant
K_hop=1.0/5.0E-12
K_LHC_RC=1.0/16.0E-12
k_params=(PSII.k_diss,PSII.k_trap,PSII.k_con,K_hop,K_LHC_RC)


minN1,maxN1=(1,100)
sig_1=sig_RC
minlp1,maxlp1=(500, 2000)
minw1,maxw1=(1.0, 10.0)
N1=round(random.uniform(minN1,maxN1))
sig_1=sig_RC
lp1=round(random.uniform(minlp1, maxlp1), 1)
w1=round(random.uniform(minw1, maxw1), 1)

s1=[N1,sig_1,lp1,w1]
branch_params=[N_b,s1]


#==============================================================================



def initialise_pop(minN1,maxN1,minlp1,maxlp1,minw1,maxw1,start_pop,gensdict,no_subunits):
    gensdict={}
    subunits=[]
    #s=[]
    #print('NO SUBS: ', no_subunits)
    for g in range(start_pop):
        s_list=[]

        for n in range (no_subunits):
            #s=[]
            N1=round(random.uniform(minN1,maxN1))
            sig_1=sig_RC
            lp1=round(random.uniform(minlp1, maxlp1), 1)
            w1=round(random.uniform(minw1, maxw1), 1)

            s = [N1, sig_1, lp1, w1]
            s_list.append(s)
            
        gensdict["g{0}".format(g+1)] = s_list
    start_gen=list(gensdict.values())
    #print('SUBUNITS:  ',gensdict)
   # print()

    return(start_gen)


fitnesses={}
def fitness(l,Ip_y,genome,N_b,s1,RC_params,k_params,branch_params,T):
    save=defaultdict(list)
    scores=[]
    efficiency=[]
    #print(genome)
    for s in genome:
        array_of_tuples = []
        branch_params=[N_b]+s
        #print('BRANCH: ',branch_params)
        out=Antenna_branched_overlap(l,Ip_y,branch_params,RC_params,k_params,T)
        nue=out['nu_e']
        phi=out['phi_F']
        scores.append(nue)
        efficiency.append(phi)
        prod=nue,phi
        for  t in s:
            array_of_tuples.append(tuple(t))
        s_tuple=tuple(array_of_tuples)
        print("Saving antenna:", array_of_tuples, prod)
        save[s_tuple].append(prod)
   
    keys_list = []
    values_list = []
    for key, values in save.items():
        keys_list.extend([key] * len(values))
        values_list.extend(values)
    fitnesses = list(zip(keys_list, values_list))

    return(fitnesses)



def selection(fitnesses,cutoff): # ok this for 
    ranked=fitnesses.copy()
    
    # select phi x nue
    products=[(key, value[0] * value[1] ) for key, value in ranked]    
    most_fit=sorted(products, key = lambda x: x[1], reverse = True)[:cutoff]
    most_keys = [item[0] for item in most_fit]
    return(most_fit, most_keys, products)



def crossover(most_keys,start_pop): 
    parents=most_keys.copy()
    split=int(cutoff/2)
    random.shuffle(parents)
    
    split = len(parents) // 2
    mum = parents[:split]
    dad = parents[split:]

    baby_genome=[]
    while len(baby_genome)<start_pop:
    #for t in range(start_pop):
        mg=random.choice(mum)  #mum    
        dg=random.choice(dad)   #dad
        chances=[]
        baby=list(mg)
        #print(baby)
        for u,(mu,du) in enumerate(zip(mg,dg)):
            chance=random.random()
            #chances.append(chance)
            if chance>=0.5:
                baby[u]=dg[u]
            clone=all(e < 0.5 for e in chances)
            if clone:
                half = len(mg) // 2
                part1_element1, part2_element1 = mg[:half], mg[half:]
                part1_element2, part2_element2 = dg[:half], dg[half:]
                #swap the corresponding parts of the elements
                swapped = part1_element1 + part2_element2
                baby_genome.append(swapped)
        baby_genome.append(baby)

    return (baby_genome)




def mutation(new_gen,mutation_rate,minN1,maxN1,minlp1,maxlp1,minw1,maxw1):
    mutated_genome = [[list(t) for t in sublist] for sublist in new_gen] # Create a copy of the original genome
    #print(mutated_genome)
    for i,j in enumerate(mutated_genome):
        for x,y in enumerate(j):
            for p,q in enumerate(list(y)):
                    if p == 0:  # N1
                        if random.random() < mutation_rate:
                            qmin = max(minN1, q - (q * 0.1))
                            qmax = min(maxN1, q + (q * 0.1))
                            y[p] = int(random.uniform(qmin, qmax))
                    elif p == 2:  # lp1
                        if random.random() < mutation_rate:
                            qmin = max(minlp1, q - (q * 0.1))
                            qmax = min(maxlp1, q + (q * 0.1))
                            y[p] = int(random.uniform(qmin, qmax))
                    elif p == 3:  # w1
                        if random.random() < mutation_rate:
                            qmin = max(minw1, q - (q * 0.1))
                            qmax = min(maxw1, q + (q * 0.1))
                            y[p] = int(random.uniform(qmin, qmax))
    return (mutated_genome)
          



def new_generation(genome,start_pop):
    ng=[]
    for p in range(start_pop):
        ng.append(random.choice(genome))

    return(ng)


def average_fitness(most_fit):
    each_score=[item[1] for item in most_fit]
    average=statistics.mean(each_score) 
    return(average)
    

def get_dfit(average1,average2):
    dfit=((average2-average1)/average1)*100   # get % change
    return(dfit)
    
    
def track_evolution(y):
    
    num_elements = len(y)
    num_sublists = len(y[0]) if y else 0
    num_values = len(y[0][0]) if y and y[0] else 0

    means=[]
    for i in range(num_sublists):
        sublist_means = []

        for j in range(num_values):
            value_sum = 0

            for element in y:
                if len(element) > i and len(element[i]) > j:
                    value_sum += element[i][j]

            sublist_mean = value_sum / num_elements if num_elements > 0 else 0
            sublist_means.append(sublist_mean)

        means.append(sublist_means)
    #print(means)
    #return (means)

    p1,p2,p3,p4=[],[],[],[]
    s1,s2,s3,s4=[],[],[],[]
    for i,j in enumerate(y):
        for x,y in enumerate(j):
            for p,q in enumerate(y):
                print('THIS IS Y; ',y)
                if p==0:    #N1
                    p1.append(y[p])
                if p==1: 
                    p2.append(y[p])
                if p==2:    # lp1
                    p3.append(y[p])
                if p==3:    # w1
                    p4.append(y[p]) 
            
                elif len(j)==1:
                    print('THIS IS J; ',j)
    
                    if x==0:    #N1
                        
                        p1.append(j[x])
                    if x==1: 
                        p2.append(j[x])
                    if x==2:    # lp1
                        p3.append(j[x])
                    if x==3:    # w1
                        p4.append(j[x]) 
               
            
    print(p1)
    pN1=statistics.mean(p1)     # find mean of each param = [x,x,x,x]:nu_e
    psig=statistics.mean(p2)
    plp1=statistics.mean(p3)
    pw1=statistics.mean(p4)
    s1=statistics.stdev(p1)
    s2=statistics.stdev(p2)
    s3=statistics.stdev(p3)
    s4=statistics.stdev(p4)
    print('STATS:')
    print(pN1,psig,plp1,pw1)
    return(means,pN1,psig,plp1,pw1,s1,s2,s3,s4)
    
    

def genetic_algorithm(N1,lp1,sig_1,w1,fitnesses,s1,branch_params,start_pop,l,Ip_y,no_subunits,T,N_b,RC_params,k_params):
    no_gen=0 
    gens=[]
    convergence=False
    ref=0
    diffs_fit=[]
    a1,a2,a3,a4=[],[],[],[]
    st1,st2,st3,st4=[],[],[],[]
    y=[]
    start_gen=initialise_pop(minN1,maxN1,minlp1,maxlp1,minw1,maxw1,start_pop,gens,no_subunits)
    no_gen=no_gen+1 
    gens.append(no_gen)
    fitnesses=fitness(l,Ip_y,start_gen,N_b,s1,RC_params,k_params,branch_params,T)
    # print('GENERATION 1 (genome:fitness):  ',fitnesses)
    most_fit,most_keys,products=(selection(fitnesses,cutoff))
    average1=average_fitness(most_fit)
    y.append(average1)
    # print('AVERAGE FITNESS FOR FITTEST GEN 1: ', average1)
    #print('MOST FIT (genomes:fitness):  ', most_fit)
    means,pN1,psig,plp1,pw1,s1,s2,s3,s4=track_evolution(most_keys)
    a1.append(pN1)
    a2.append(psig)
    a3.append(plp1)
    a4.append(pw1)
    st1.append(s1)
    st2.append(s2)
    st3.append(s3)
    st4.append(s4)
 
  
    while convergence==False:
 
        new_genomes=mutation(crossover(most_keys,start_pop), mutation_rate, minN1, maxN1, minlp1, maxlp1, minw1, maxw1) 
        fitnesses=fitness(l,Ip_y,new_genomes,N_b,s1,RC_params,k_params,branch_params,T) 
                
        no_gen=no_gen+1 
        gens.append(no_gen)
        most_fit,most_keys,products=(selection(fitnesses,cutoff))
        average2=average_fitness(most_fit)
        y.append(average2)    
        dfit=get_dfit(average1,average2)
        #print('MOST FIT (genomes:fitness):  ', most_fit)
        means,pN1,psig,plp1,pw1,s1,s2,s3,s4=track_evolution(most_keys)
        #print(gauss(lp_RC,w_RC,plp1,pw1))
        a1.append(pN1)
        a2.append(psig)
        a3.append(plp1)
        a4.append(pw1)
        st1.append(s1)
        st2.append(s2)
        st3.append(s3)
        st4.append(s4)
        dfit=get_dfit(average1,average2)
        # print('FITNESS DIFFERENCE FROM LAST GENERATION: ',dfit,'%')
        diffs_fit.append(dfit)
        average1=average2
        nuephi=find_fitness(most_keys,fitnesses)
        meannue,meanphis=get_means(nuephi)
        #print(dnu_e,most_keys)
        if abs(dfit-ref) < 0.01:
            convergence=True
            print('Convergence = True')
            print('convergence/most fit genomes:')
            print(dfit,most_keys)
            print()
            print('MEAN PARAMS: ',means)
            print()
            
            nuephi=find_fitness(most_keys,fitnesses)
            meannue,meanphis=get_means(nuephi)
            print('MEAN NUE + PHI: ',meannue,meanphis)
            #SAVE LAST PARAMS
            # params=[pN1,psig,plp1,pw1]
            # print('LAST AVG PARAMS: ',params)
            # print(a1,a3,a4)
            with open('{}.txt'.format(thing), 'w') as file:
                for item in means:
                    for i in item:
                        file.write("%s\n" % str(i))
                    file.write("%s\n" % str(meannue))
                    file.write("%s\n" % str(meanphis))
            file.close()
            
        if no_gen>=150:
            print('MOST KEYS; ',most_keys)
            break
    
    return(means,y,gens,nuephi,a1,a2,a3,a4,st1,st2,st3,st4)
   
    

def find_fitness(most_keys,fitnesses):
    nuephi=[]
    for m in most_keys:
        for f in fitnesses:
            if m==f[0]:
                item=(m,f[1])
                nuephi.append(item)
                break
            
    return(nuephi)


def get_means(nuephi):
    nues=[]
    phis=[]
    for i in nuephi:
        for o in i[1]:
            nues.append(i[1][0])
            phis.append(i[1][1])
    meannue=statistics.mean(nues)
    meanphis=statistics.mean(phis)
    return(meannue,meanphis)
            


    



for v in range(1,vers):
    thing='{}.b{}.s{}.v{}'.format(spec,N_b,no_subunits,v)
    means,y,gens,nuephi,a1,a2,a3,a4,st1,st2,st3,st4=genetic_algorithm(N1,lp1,sig_1,w1,fitnesses,s1,branch_params,start_pop,l,Ip_y,no_subunits,T,N_b,RC_params,k_params)



# plt.plot(gens,y, color='navy', linewidth=0.5,label='Combined fitness')
# plt.plot(gens,a1, color='darkred', linewidth=0.5,label='Avg. number of pigments')
# plt.plot(gens,a3, color='darkgreen', linewidth=0.5,label='Avg. peak absorbance wavelength')
# plt.plot(gens,a4, color='orange', linewidth=0.5,label='Avg. subunit width')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 8), gridspec_kw={'height_ratios': [1,1,1]})
plt.suptitle(r'Antenna Properties over Generations',fontsize=13, fontweight="bold", y=0.91)

ax1.plot(gens, a3, label="Avg. peak absorbance wavelength",linewidth=0.7,color='darkgreen')

ax2.plot(gens, a1, label="Avg. number of pigments", color="red", linewidth=0.7)
ax2.plot(gens, a4, label="Avg. subunit width", color="orange",linewidth=0.7)
ax3.plot(gens, y, label="Combined fitness", color="navy",linewidth=0.7)

ax1.set_ylabel('Wavelength (nm)')
ax2.set_ylabel('Arbitrary units')
ax3.set_ylabel(r'$\nu_e\phi_e$')

d = 0.015  
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)  
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  


kwargs.update(transform=ax2.transAxes)  
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  

kwargs.update(transform=ax3.transAxes)
ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  

ax1.legend()
ax2.legend()


#plt.yscale('log')
plt.xlabel("Generation")
#plt.ylabel('Arbitrary log scale')                 #r"Average $\nu_e\phi_e$")
plt.legend()
plt.show()













