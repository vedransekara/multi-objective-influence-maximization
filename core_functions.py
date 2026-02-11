import random
from collections import defaultdict
import operator
import networkx as nx

def ICM(network,seeds_,p_infection):
    '''
    Independent cascade model
    ------------------    
    Input:
    * network: directed opr undirected network
    * seeds: list of seed nodes or number of initial seeds to simulate the dynamics from
    * p: infection/spreading probability 
    ------------------
    Output: 
    * for activated node, timestamp of when they were activated
    ------------------
    '''
    if type(seeds_) != list:
        infected = set(random.sample(sorted(network.nodes()),seeds_)) # infect seeds
    else:
        infected = set(seeds_)
    
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
            for neighbor in network.neighbors(node):
                if (random.random() < p_infection) and (neighbor not in activated):
                    # register activation time
                    activation_time[neighbor] = t

                    # update sets
                    new_infected.add(neighbor)
                    activated.add(neighbor) # add node here to prevent newly infected nodes to be infected again

        # set of newly infected nodes
        infected = new_infected.copy()
        
        # update time counter
        t += 1
    
    # return list of infected/recovered nodes
    return activation_time

def load_network(path_):
    '''
    Loads empirical networks from files
    Removes self-loops, and if network has multiple components, it returns only the lagest connected component
    ------------------    
    Input:
    * path_: path + file name of the network to load
    ------------------
    Output: 
    * a networkx graph
    ------------------
    '''
    
    H = nx.Graph()
    with open(path_) as f:
        for line in f:
            try:
                i,j = map(int,line.strip().split()) # everything else
            except ValueError:
                i,j,w = map(int,line.strip().split()) # URV, Cond-mat
            H.add_edge(i,j)
    
    # remove selfedges
    H.remove_edges_from(nx.selfloop_edges(H))
    
    return max([H.subgraph(c) for c in nx.connected_components(H)],key=len)

def load_pc(path_):
    '''
    Loads the critical infection probability for a network
    ------------------    
    Input:
    * path_: path + filename of the pickle file to load
    ------------------
    Output: 
    * returns a float value for the critical infection probability
    ------------------
    '''

    import pickle
    
    # loads the 
    pkl_file = open(path_,'rb')
    Sp = pickle.load(pkl_file,encoding='latin1')
    pkl_file.close()
    
    # estimate parameters
    return round(max(Sp,key=operator.itemgetter(1))[0],6)

def coreHD(H,no_seeds):
    '''
    Run the CoreHD heuristic on the network H. Finds the no_seeds must influential nodes
    ------------------    
    Input:
    * H: undirected and unweighted networkx graph
    * no_seeds: integer for how many seeds nodes you want the heuristic to return
    ------------------
    Output: 
    * returns a list of seeds nodes
    ------------------
    '''
    
    seed_nodes = []
    for i in range(no_seeds):
        
        # find k=2 core
        kcore = nx.k_core(H,k=2)
        
        # find node or nodes with max degree node
        k_max = max(list(zip(*kcore.degree()))[1])
        
        # if mode candidates pick random node
        node = random.choice([n for n,k in kcore.degree() if k == k_max])
                    
        H.remove_node(node)
        seed_nodes.append(node)

    return seed_nodes

def degree_discount(H,p_infection,no_seeds):
    '''
    Run the degree discount heuristic on the network H. Finds the no_seeds must influential nodes
    ------------------    
    Input:
    * H: undirected and unweighted networkx graph
    * p_infection: the degree discount heuristic uses a message passing framework and requires a transmssion/infection probability
    * no_seeds: integer for how many seeds nodes you want the heuristic to return
    ------------------
    Output: 
    * returns a list of seeds nodes
    ------------------
    '''
    # initialize stuff
    seed_nodes = set()
    dd_v = dict(H.degree())
    t_v = dict([(n,0) for n in H.nodes()])

    # select nodes
    for i in range(no_seeds):
        k_max = max([k for n,k in dd_v.items() if n not in seed_nodes])
        # randomly select one node that has k = k_max
        node = random.choice([n for n,k in dd_v.items() if (k == k_max) and (n not in seed_nodes)])
        seed_nodes.add(node)

        # compute t_v abnd update dd_v-
        for neigh in H.neighbors(node):
            t_v[neigh] += 1

            d = H.degree(neigh)
            t = t_v[neigh]

            dd_v[neigh] = d - 2*t - (d-t)*t*p_infection
    return list(seed_nodes)

def highest_degree(H,no_seeds):
    '''
    Run the highest degree heuristic on the network H. Finds the no_seeds must influential nodes
    ------------------    
    Input:
    * H: undirected and unweighted networkx graph
    * no_seeds: integer for how many seeds nodes you want the heuristic to return
    ------------------
    Output: 
    * returns a list of seeds nodes
    ------------------
    '''
    # find the minimum value of k for the number of seed nodes
    k_min = min(sorted(list(zip(*H.degree()))[1],reverse=True)[:no_seeds])
    # select all nodes with k above k_min
    high_k_nodes = [n for n,k in H.degree() if k > k_min]
    # pick all nodes with k = k_min
    k_min_nodes = [n for n,k in H.degree() if k == k_min]
    # calculate how many nodes we need to pick from the set of nodes with k = k_min
    sample_size = no_seeds - len(high_k_nodes)
    # pick randomly from list of nodes wth k_min_nodes to meet seed size
    seed_nodes = high_k_nodes + random.sample(k_min_nodes,sample_size)
    
    return seed_nodes

def k_core(H,no_seeds):
    '''
    Run the kcore heuristic on the network H. Finds the no_seeds must influential nodes
    ------------------    
    Input:
    * H: undirected and unweighted networkx graph
    * no_seeds: integer for how many seeds nodes you want the heuristic to return
    ------------------
    Output: 
    * returns a list of seeds nodes
    ------------------
    '''
    # select seed nodes
    # find the minimum value of k for the number of seed nodes
    core_min = sorted(nx.core_number(H).values(),reverse=True)[no_seeds]
    # select all nodes with core number above core_min
    high_core_nodes = [n for n,k in nx.core_number(H).items() if k > core_min]
    # select all nodes with core_number = core_min
    core_min_nodes = [n for n,k in nx.core_number(H).items() if k == core_min]
    # figure our how many nodes we are missing to add to readch required set size
    sample_size = no_seeds - len(high_core_nodes)
    # sample nodes that have k = k_min randomly from k_min_nodes to meet the required set size 
    seed_nodes = high_core_nodes + random.sample(core_min_nodes,sample_size)

    return seed_nodes