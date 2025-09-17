import networkx as nx
import array
import argparse
import numpy as np
import pandas as pd
from deap import algorithms
import random
import operator
import math
from collections import Counter, defaultdict
from deap import creator
from deap import base
from deap import tools
import multiprocessing

###Global vars####
heus = ["HD","DD","KC","AQ","CHD","RD","MIX","FILE","FILE2","IDR","IDR2"]
networks = ["G","SP"]
algos = ["ICM","NSGA3","PROB","EAS"]
muts = ["SET","TABU","MIX"]

def readSeedsFile(fn):
    f = open(fn)
    L = []
    line = f.readline().strip()
    while len(line)>0:
        words = line.split()
        wordsi = []
        for i in range(len(words)):
            wordsi.append(int(words[i]))
        L.append(wordsi)
        line = f.readline().strip()
    f.close()
    return L

def readSeedsFile2(fn):
    f = open(fn)
    L = []
    R = []
    V = []
    line = f.readline().strip()
    j = 0
    while len(line)>0:
        print(j)
        j += 1
        lineS = line.split("(")
        words = lineS[0].strip()[1:-1].split(",")
        words2 = lineS[1].strip()[:-1].split(",")
        wordsi = []
        k=0
        for i in range(len(words)):
            wordsi.append(int(words[i].strip()))
            k+=1

        L.append(wordsi)
        R.append(float(words2[0].strip()))
        V.append(float(words2[1].strip()))
        line = f.readline().strip()
    f.close()
    return L,R,V

def readSPSfile(fn):
    sps = {}
    AvgP = {}
    AvgPaux = {}

    f = open(fn)
    line = f.readline().strip()
    while line[0] != "*":
        words = line.split("-")
        k = int(words[0].strip())
        sps[k] = {}
        words2 = words[1].split(",")
        for i in range(len(words2)):
            words3 = words2[i].split(":")
            l = int(words3[0])
            spskl = int(words3[1])
            sps[k][l] = spskl
    line = f.readline().strip()
    while line[0] != "*":
        words = line.split()
        AvgP[int(words[0])] = float(words[1])
    line = f.readline().strip()
    while len(line) > 0:
        words = line.split()
        AvgPaux[int(words[0])] = float(words[1])

    f.close()

    return sps,AvgP,AvgPaux

#load network
def load_network(fn):
    G = nx.Graph()
    with open(fn) as f:
        for line in f:
            words = line.split()
            i = int(words[0])
            j = int(words[1])
            G.add_edge(i, j)
    return G

# load pc file
def load_pc(fn):
    pkl_file = open(fn, 'r')
    #Sp = pickle.load(fn)
    pkl_file.close()
    Sp = pd.read_pickle(fn)
    # estimate parameters
    return round(max(Sp, key=operator.itemgetter(1))[0], 3)


# Highest degree
def HD(G,seeds):
    # find the minimum value of k for the number of seed nodes
    k_min = min(sorted(dict(G.degree()).values(), reverse=True)[:seeds])
    # select all nodes with k above k_min
    high_k_nodes = [n for n, k in dict(G.degree()).items() if k > k_min]
    # select all nodes with k = k_min
    k_min_nodes = [n for n, k in dict(G.degree()).items() if k == k_min]
    # nodes we need randomly pick from k_min_nodes to meet seed size
    sample_size = seeds - len(high_k_nodes)
    # select seed nodes and update seed statistics
    seed_nodes_ = high_k_nodes + random.sample(k_min_nodes, sample_size)

    return seed_nodes_

# degree discount
def degree_discount(G,seeds,p_infection):
    # initializse stuff
    seed_nodes = set()
    dd_v = dict(G.degree())
    t_v = dict([(n,0) for n in G.nodes()])

    # select nodes
    for i in range(seeds):
        k_max = max([k for n,k in dd_v.items() if n not in seed_nodes])
        # randomly select one node that has k = k_max
        node = random.choice([n for n,k in dd_v.items() if (k == k_max) and (n not in seed_nodes)])
        seed_nodes.add(node)

        # compute t_v abnd update dd_v
        for neigh in G.neighbors(node):
            t_v[neigh] += 1

            d = G.degree(neigh)
            t = t_v[neigh]

            dd_v[neigh] = d - 2*t - (d-t)*t*p_infection
    return list(seed_nodes)

# core HD
def coreHD(G,seeds):
    H = G.copy().copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    seed_nodes = []
    for i in range(seeds):
        # find k=2 core
        kcore = nx.k_core(H, k=2)

        # find node or nodes with max degree node
        k_max = max(dict(kcore.degree()).values())

        # if mode candidates pick random node
        node = random.choice([n for n, k in dict(kcore.degree()).items() if k == k_max])

        H.remove_node(node)
        seed_nodes.append(node)

    return seed_nodes

# k-core
def kCore(G,seeds):
    H = G.copy().copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    # select seed nodes
    # find the minimum value of k for the number of seed nodes
    core_min = sorted(nx.core_number(H).values(), reverse=True)[seeds]
    # select all nodes with core number above core_min
    high_core_nodes = [n for n, k in nx.core_number(H).items() if k > core_min]
    # select all nodes with core_number = core_min
    core_min_nodes = [n for n, k in nx.core_number(H).items() if k == core_min]
    # nodes we need randomly pick from k_min_nodes to meet seed size
    sample_size = seeds - len(high_core_nodes)
    # select seed nodes and update seed statistics
    seed_nodes_ = high_core_nodes + random.sample(core_min_nodes, sample_size)

    return seed_nodes_

# Acquaintance
def acquaintance(G,seeds):
    sample = random.sample(G.nodes(), seeds)
    seed_nodes_ = []
    for n in sample:
        neis = [k for k in G.neighbors(n) if k not in seed_nodes_]
        if len(neis)==0:
            return sample
        rk = random.choice(neis)
        while rk in seed_nodes_:
            rk = random.choice(neis)
        seed_nodes_.append(rk)
    #seed_nodes_ = [random.choice(G.neighbors(n)) for n in random.sample(G.nodes(), seeds)]

    return seed_nodes_
    
def randomSeeds(G,seeds):
    return random.sample(G.nodes(), seeds)


def ICM(G, seed_nodes_, p_infection):
    '''
        Independent cascade model
        ------------------
        Input:
        * G: directed opr undirected network
        * seed_nodes_: list of seed nodes or number of initial seeds to simulate the dynamics from
        * p: infection/spreading probability
        ------------------
        Output:
        * for activated node, timestamp of when they were activated
        ------------------
        '''
    # if type(seeds) != list:
    # infected = set(random.sample(network.nodes(),seeds)) # infect seeds
    # else:
    infected = set(seed_nodes_)

    activated = infected.copy()
    activation_time = defaultdict(int)

    # add initial seed to activated & activation_time
    for n in infected:
        activation_time[n] = 0

    t = 1
    # run infection process
    while infected != set():
        new_infected = set()
        for node in infected:
            # print node
            for neighbor in G.neighbors(node):
                # print "--- ",neighbor
                r = random.random()
                # print "--- ",r
                if (r < p_infection) and (neighbor not in activated):
                    # register activation time
                    activation_time[neighbor] = t

                    # update sets
                    new_infected.add(neighbor)
                    # print "--- infected"
                    activated.add(neighbor)  # add node here to prevent newly infected nodes to be infected again

        # set of newly infected nodes
        infected = new_infected.copy()

        # update time counter
        t += 1

    # return list of infected/recovered nodes
    return activation_time

#Generate random seeds
def generate(n, seeds):
    sol = np.zeros(n + 1)

    for i in range(seeds):
        rn = random.randint(1, n)
        while sol[rn] == 1:
            rn = random.randint(1, n)
        sol[rn] = 1
    return sol

##############RANDOM SIMULATION################
def randomSimulation(G,seeds,pc):

    random_activation_count = Counter()
    activation_time = defaultdict(Counter)
    seed_stat = Counter()

    random.seed()
    n = G.number_of_nodes()
    m = n*10

    # run multiple spreading processes
    for repetition in range(m):
        rdseeds = generate(n, seeds)
        seed_nodes_ = []
        for i in range(n):
            if rdseeds[i] == 1:
                seed_nodes_.append(i)
        for repetition2 in range(1):

            seed_stat += Counter(seed_nodes_)

            # run spreading dynamics
            c = ICM(G, seed_nodes_, pc)

            # update time counter
            for n_, time_ in c.items():
                activation_time[n_][time_] += 1

            # update activation counter
            random_activation_count += Counter(c.keys())

    return random_activation_count


##############SOLUTION SIMULATION################
def fixedSimulation(G, args,pc):
    activation_count = Counter()
    activation_time = defaultdict(Counter)
    seed_stat = Counter()

    random.seed()
    n = G.number_of_nodes()
    m = n * 10

    # run multiple spreading processes
    for repetition in range(m):
        if (args.Heu == "DD"):
            seed_nodes_ = degree_discount(G,seeds,p_infection)
        if (args.Heu == "HD"):
            seed_nodes_ = HD(G,seeds)
        if (args.Heu == "KC"):
            seed_nodes_ = kCore(G,seeds)
        if (args.Heu == "CHD"):
            seed_nodes_ = coreHD(G,seeds)
        if (args.Heu == "AQ"):
            seed_nodes_ = acquaintance(G,seeds)

        seed_stat += Counter(seed_nodes_)

        # run spreading dynamics
        c = ICM(G, seed_nodes_, pc)

        # update time counter
        for n_, time_ in c.items():
            activation_time[n_][time_] += 1

        # update activation counter
        activation_count += Counter(c.keys())

    return activation_count

#Init individuals

def load_individuals(icls,n,args, G, seeds, pc, AvgPaux,sps):

    population = []
    if(args.Heu=="IDR"):
        seed_nodes_ = degree_discount(G, seeds, pc)
        seed_nodes_ = icls(seed_nodes_)
        if(len(seed_nodes_)!=seeds):
            print("DD ERROR!!!!")
            exit(-1)
        population.append(seed_nodes_)
        seed_nodes_ = HD(G, seeds)
        seed_nodes_ = icls(seed_nodes_)
        if(len(seed_nodes_)!=seeds):
            print("HD ERROR!!!!")
            exit(-1)
        population.append(seed_nodes_)
        seed_nodes_ = kCore(G, seeds)
        seed_nodes_ = icls(seed_nodes_)
        if(len(seed_nodes_)!=seeds):
            print("KC ERROR!!!!")
            exit(-1)
        population.append(seed_nodes_)
        seed_nodes_ = coreHD(G, seeds)
        seed_nodes_ = icls(seed_nodes_)
        if(len(seed_nodes_)!=seeds):
            print("CHD ERROR!!!!")
            exit(-1)
        population.append(seed_nodes_)
        seed_nodes_ = acquaintance(G, seeds)
        seed_nodes_ = icls(seed_nodes_)
        if(len(seed_nodes_)!=seeds):
            print("AQ ERROR!!!!")
            exit(-1)
        population.append(seed_nodes_)


        for i in range(n-5):
            seed_nodes_ = randomSeeds(G, seeds)
            seed_nodes_ = icls(seed_nodes_)
            if(len(seed_nodes_)!=seeds):
                print("RD ERROR!!!!")
                exit(-1)
            population.append(seed_nodes_)

        return population

    if (args.Heu == "IDR2"):
        for i in range(int(n/2)):
            rd = random.randrange(0, 5)
            if (rd == 0):
                seed_nodes_ = degree_discount(G, seeds, pc)
            if (rd == 1):
                seed_nodes_ = HD(G, seeds)
            if (rd == 2):
                seed_nodes_ = kCore(G, seeds)
            if (rd == 3):
                seed_nodes_ = coreHD(G, seeds)
            if (rd == 4):
                seed_nodes_ = acquaintance(G, seeds)

            seed_nodes_ = icls(seed_nodes_)
            population.append(seed_nodes_)

        for i in range(n-int(n/2)):
            seed_nodes_ = randomSeeds(G, seeds)
            seed_nodes_ = icls(seed_nodes_)
            population.append(seed_nodes_)

    for i in range(n):
        if (args.Heu == "DD"):
            seed_nodes_ = degree_discount(G, seeds, pc)
        if (args.Heu == "HD"):
            seed_nodes_ = HD(G, seeds)
        if (args.Heu == "KC"):
            seed_nodes_ = kCore(G, seeds)
        if (args.Heu == "CHD"):
            seed_nodes_ = coreHD(G, seeds)
        if (args.Heu == "AQ"):
            seed_nodes_ = acquaintance(G, seeds)
        if (args.Heu == "RD"):
            seed_nodes_ = randomSeeds(G, seeds)

        if(args.Heu == "MIX"):
            rd = random.randrange(0,6)
            if (rd == 0):
                seed_nodes_ = degree_discount(G, seeds, pc)
            if (rd == 1):
                seed_nodes_ = HD(G, seeds)
            if (rd == 2):
                seed_nodes_ = kCore(G, seeds)
            if (rd == 3):
                seed_nodes_ = coreHD(G, seeds)
            if (rd == 4):
                seed_nodes_ = acquaintance(G, seeds)
            if (rd == 5):
                seed_nodes_ = randomSeeds(G, seeds)

            
        seed_nodes_ = icls(seed_nodes_)
        
        population.append(seed_nodes_)

    return population

def calculateAvgP(G,seeds,pc,MAXPLEN):
    AvgPaux = {}
    nodes = G.nodes()
    n = len(nodes)
    sps = {}
    for k in nodes:
        sps[k] = nx.single_source_shortest_path_length(G, source=k, cutoff = MAXPLEN)
        for node in sps[k]:
            if node not in AvgPaux:
                AvgPaux[node] = 0.0
            AvgPaux[node] += pow(pc, sps[k][node])
    AvgP ={}
    for k in G.nodes():
        AvgPaux[k] = (AvgPaux[k]/float(n-1))
        AvgP[k] = 1.0 - pow(1.0-AvgPaux[k],seeds)

    return sps,AvgP,AvgPaux



def evaluateSP(ind,G,pc,avgP,sps):

    R = {}
    for k in G.nodes():
        R[k] = 1.0
        if k not in ind:
            for l in ind:
                if k in sps[l]:
                    R[k] = R[k]*(1.0-pow(pc,sps[l][k]))
            R[k] = 1.0 - R[k]
    Reach = 0.0
    for k in G.nodes():
        Reach += R[k]

    Vulnerable = 0
    for k in G.nodes():
        if R[k]<avgP[k]:
            Vulnerable += 1

    return Reach,Vulnerable


def calculateRaux(ind, G, pc, sps):

    R = {}
    for k in G.nodes():
        R[k] = 1.0
        if k not in ind:
            for l in ind:
                if k in sps[l]:
                    R[k] = R[k] * (1.0 - pow(pc,sps[l][k]))
    return R

def getVfromRaux(ind,node,Raux,pc,avgP,sps):
    Vulnerable = 0
    R = Raux.copy()
    for k in R:
        if k not in ind:
            if k in sps[node]:
                R[k] = R[k] * (1.0 - pow(pc,sps[node][k]))
                R[k] = 1.0 - R[k]
            else:
                R[k] = 1.0 - R[k]

            if R[k] < avgP[k]:
                Vulnerable += 1

    return Vulnerable

def cxSet(ind1, ind2, seeds):
    """Apply a crossover operation on input
    """

    temp = set(ind1|ind2)                # Used in order to keep type
    ind1.clear()
    rks = random.sample(tuple(temp),seeds)
    for rk in rks:
        ind1.add(rk)

    ind2.clear()
    rks = random.sample(tuple(temp), seeds)
    for rk in rks:
        ind2.add(rk)
        
    if(len(ind1)!=seeds):
        print("CXTSET IND1 ERROR!!!!")
    if(len(ind2)!=seeds):
        print("CXTSET IND2 ERROR!!!!")

    return ind1, ind2

def mutSet(individual,nodes,seeds):

    nodesaux = nodes.copy()
    for node in individual:
        nodesaux.remove(node)
    size = int(math.ceil(seeds/10.0))
    rks = random.sample(tuple(individual),size)
    for rk in rks:
        individual.remove(rk)
    rks = random.sample(tuple(nodesaux),size)

    for rk in rks:
        individual.add(rk)
        
    if(len(individual)!=seeds):
        print("MUTSET ERROR!!!!")

    return individual,

def mutSetMix(individual,nodes,G,pc,avgP,sps,size,seeds,freq):
    rd = random.random()
    if rd<freq:
        return mutSetTabu(individual,nodes,G,pc,avgP,sps,size)
    return mutSet(individual,nodes,seeds)


def mutSetTabu(individual,nodes,G,pc,avgP,sps,size):

    rk = random.choice(tuple(individual))
    individual.remove(rk)

    bestV = 1000000
    bestNode = -1
    ind = individual.copy()
    Raux = calculateRaux(ind,G,pc,sps)
    sampleNodes = random.sample(tuple(nodes),min(size,len(nodes)))
    for node in sampleNodes:
        if node not in individual:
            ind.add(node)
            V = getVfromRaux(ind,node,Raux,pc,avgP,sps)
            if V < bestV:
                bestV = V
                bestNode = node
            ind.remove(node)

    if bestNode!=-1:
        individual.add(bestNode)
    else:
        print("MUTSETTABU ERROR!!!!")

    return individual,

if __name__ == '__main__':
    # Set up input arguments
    p = argparse.ArgumentParser()
    p.add_argument("-NetFile", type=str, default="", help='File with network',required=True)
    p.add_argument("-PcFile", type=str, default="", help='File with prob of infection',required=True)
    p.add_argument("-SeedsFile", type=str, default="", help='File with seeds to test fitness - use with Heu=FILE or FILE2')
    p.add_argument("-SpsFile", type=str, default="", help='File with seeds to test fitness - use with Heu=FILE2')
    p.add_argument("-Heu", type=str, default="MIX", help='Heuristic to select seeds', choices=heus)
    p.add_argument("-Algo", type=str, default="NSGA3", help='Algorithm', choices=algos)
    p.add_argument("-MUTOP", type=str, default="SET", help='Mutation Operator', choices=muts)
    p.add_argument("-Network", type=str, default="SP", help='Network type for eval', choices=networks)
    p.add_argument("-MU", type=int, default=100, help='Population size')
    p.add_argument("-LAMBDA", type=int, default=100, help='Population size for offspring (for EAS)')
    p.add_argument("-NGEN", type=int, default=20, help='Number of generations')
    p.add_argument("-CXPB", type=float, default=0.8, help='Crossover probability')
    p.add_argument("-MUTPB", type=float, default=0.8, help='Mutation probability')
    p.add_argument("-TABUFREQ", type=float, default=0.01, help='Frequency of tabu mutation in mixed operator')
    p.add_argument("-NEIGSIZE", type=int, default=100, help='Neighborhood size for tabu mutation in %% of nodes')
    p.add_argument("-MAXPLEN", type=int, default=10, help='Max shortest path length considered')
    p.add_argument("-seed", type=int, default=-1, help='seed')
    args = p.parse_args()

    # load network
    print("Load Network")
    G = load_network(args.NetFile)
    m = 10 * len(G)

    # number of seeds (depends on network size)
    seeds = int(round(1 * len(G) / 100., 0))  # 1% of nodes in network
    if seeds<4:
        seeds = 4
    # load p_c value
    print("Load infection probability")
    p_infection = 1 * load_pc(args.PcFile)

    if args.seed==-1:
        random.seed()
    else:
        random.seed(args.seed)

    if(args.Algo=="ICM"):
        print("ICM")
        if (args.Network == "G"):
            random_activation_count = randomSimulation(G,seeds,p_infection)
            activation_count = fixedSimulation(G,args,p_infection)

        totalnodes = 0
        avgnodes = 0
        for node in activation_count:
            totalnodes += 1
            avgnodes += activation_count[node]

        print('total nodes ', totalnodes)
        print('average nodes ', avgnodes / float(m))
        bavg = 0
        n = len(activation_count)
        for key in activation_count:
            if activation_count[key] < random_activation_count[key]:
                bavg += 1
        print('above avg', n - bavg)
        print('cascade size', (avgnodes / float(m)) / n)
        print('vulnerable', bavg / float(n))

    else:

        
        if(args.Algo=="PROB"):
            print("PROBS")
            
            if(args.Heu=="FILE"):
                L = readSeedsFile(args.SeedsFile)
                if (args.Network == "SP"):
                    sps, avgP, avgPaux = calculateAvgP(G, seeds, p_infection, args.MAXPLEN)
                for i in range(len(L)):
                    seed_nodes_ = L[i]
                    if (args.Network == "SP"):
                        R,V = evaluateSP(seed_nodes_,G,p_infection,avgP,sps)
                    print(seed_nodes_,R,V)
            elif(args.Heu=="FILE2"):
                #sps, avgP, avgPaux = readSPSfile(args.SpsFile)
                L,Rold,Vold = readSeedsFile2(args.SeedsFile)
                if (args.Network == "SP"):
                    sps, avgP, avgPaux = calculateAvgP(G, seeds, p_infection, args.MAXPLEN)
                for i in range(len(L)):
                    seed_nodes_ = L[i]
                    if (args.Network == "SP"):
                        R, V = evaluateSP(seed_nodes_, G, p_infection, avgP, sps)
                    print(seed_nodes_, R, V,Rold[i],Vold[i])
            else:
                seed_nodes_ = []
                if (args.Heu == "DD"):
                    seed_nodes_ = degree_discount(G, seeds, pc)
                if (args.Heu == "HD"):
                    seed_nodes_ = HD(G, seeds)
                if (args.Heu == "KC"):
                    seed_nodes_ = kCore(G, seeds)
                if (args.Heu == "CHD"):
                    seed_nodes_ = coreHD(G, seeds)
                if (args.Heu == "AQ"):
                    seed_nodes_ = acquaintance(G, seeds)
                if (args.Network == "SP"):
                    R,V = evaluateSP(seed_nodes_,G,p_infection,avgP)
                print('seeds',seed_nodes_)
                print('cascade size',R)
                print('vulnerable',V)
        elif(args.Algo=="NSGA3"):
            if (args.Network == "SP"):
                sps, avgP, avgPaux = calculateAvgP(G, seeds, p_infection, args.MAXPLEN)

            print("NSGA3")
            MU = args.MU
            NGEN = args.NGEN
            CXPB = args.CXPB
            MUTPB = args.MUTPB

            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # cascade size and vulnerability
            creator.create("Individual", set, fitness=creator.FitnessMulti)

            toolbox = base.Toolbox()

            toolbox.register("population", load_individuals, creator.Individual, args=args, G=G, seeds=seeds,
                             pc=p_infection, AvgPaux=avgPaux, sps=sps)

            if (args.Network=="SP"):
                toolbox.register("evaluate", evaluateSP, G=G,pc=p_infection,avgP=avgP,sps=sps)

            toolbox.register("mate", cxSet, seeds=seeds)
            if(args.MUTOP=="SET"):
                toolbox.register("mutate", mutSet,nodes=set(G.nodes()),seeds=seeds)
            elif(args.MUTOP=="TABU"):
                size = int(len(G.nodes())*(args.NEIGSIZE/100.0))
                toolbox.register("mutate", mutSetTabu, nodes=set(G.nodes()), G=G,pc=p_infection,avgP=avgP,sps=sps,size=size)
            elif (args.MUTOP == "MIX"):
                size = int(len(G.nodes()) * (args.NEIGSIZE / 100.0))
                toolbox.register("mutate", mutSetMix, nodes=set(G.nodes()), G=G, pc=p_infection, avgP=avgP, sps=sps,
                                 size=size,seeds=seeds,freq=args.TABUFREQ)
            toolbox.register("select", tools.selNSGA2)

            # Initialize statistics object
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
            

            logbook = tools.Logbook()
            logbook.header = "gen", "evals", "std", "min", "avg", "max"

            print("Init population")
            pop = toolbox.population(n=MU)
            print("Init HoF")
            hof = tools.ParetoFront()

            # Evaluate the individuals with an invalid fitness
            print("Evaluate individuals")
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Compile statistics about the population
            print("compile record")
            record = stats.compile(pop)
            logbook.record(gen=0, evals=len(invalid_ind), **record)
            print(logbook.stream)
            hof.update(pop)
            # Begin the generational process
            for gen in range(1, NGEN):

                offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Select the next generation population from parents and offspring
                pop = toolbox.select(pop + offspring, MU)
                hof.update(pop)
                # Compile statistics about the new population
                record = stats.compile(pop)
                logbook.record(gen=gen, evals=len(invalid_ind), **record)
                print(logbook.stream)


            for ind in hof:
                print(sorted(ind), ind.fitness.values)


        elif (args.Algo == "EAS"):


            if (args.Network == "SP"):
                sps, avgP, avgPaux = calculateAvgP(G, seeds, p_infection, args.MAXPLEN)

            print("EAS")
            MU = args.MU
            NGEN = args.NGEN
            CXPB = args.CXPB
            MUTPB = args.MUTPB
            LAMBDA = args.LAMBDA

            creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # cascade size and vulnerability
            creator.create("Individual", set, fitness=creator.FitnessMulti)

            toolbox = base.Toolbox()
            toolbox.register("population",load_individuals, creator.Individual,  args=args, G=G, seeds=seeds, pc=p_infection, AvgPaux=avgPaux, sps=sps)

            if (args.Network == "SP"):
                toolbox.register("evaluate", evaluateSP, G=G, pc=p_infection, avgP=avgP,sps=sps)

            toolbox.register("mate", cxSet, seeds=seeds)
            if (args.MUTOP == "SET"):
                toolbox.register("mutate", mutSet, nodes=set(G.nodes()), seeds=seeds)
            elif (args.MUTOP == "TABU"):
                size = int(len(G.nodes()) * (args.NEIGSIZE / 100.0))
                toolbox.register("mutate", mutSetTabu, nodes=set(G.nodes()), G=G, pc=p_infection, avgP=avgP, sps=sps,
                                 size=size)
            elif (args.MUTOP == "MIX"):
                size = int(len(G.nodes()) * (args.NEIGSIZE / 100.0))
                toolbox.register("mutate", mutSetMix, nodes=set(G.nodes()), G=G, pc=p_infection, avgP=avgP, sps=sps,
                                 size=size, seeds=seeds, freq=args.TABUFREQ)
            toolbox.register("select", tools.selNSGA2)

            pop = toolbox.population(n=MU)
            hof = tools.ParetoFront()
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)

            algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                                      halloffame=hof,verbose=True)

            for ind in hof:
                print(sorted(ind),ind.fitness.values)


