import numpy as np
import networkx as nx

def prepare_init_state(N, I0):
    """
    Prepare a system of N nodes and I0 infected.
    
    Parameters
    ----------
    N : int, number of nodes
    I0 : int, number of initial infected nodes
    
    Returns
    -------
    state :  numpy array of shape (N,3) 
        state[:,0] = 1 for the susceptible, 0 for the others
        state[:,1] = 1 for the infected, 0 for the others
        state[:,2] = 1 for the recovered, 0 for the others
    """
    susceptible = np.ones(N) 
    seeds = np.random.choice(np.arange(N), size = I0)
    susceptible[seeds] = 0
    state = np.zeros((N,3))
    state[:,0] = susceptible
    state[seeds,1] = 1
    return state



def SIRS_step(A, state, beta, mu, gamma, T=0.5, debug=False):
    """
    SIRS step for a single network. Updated A and state needs to be computed
    before calling this function (take into account mobility + dynamic contacts).
    Works with synchronous update (e.g. new infected cannot recover in this step).
    
    Parameters
    ----------
    A : numpy matrix, adjacency matrix
    state : numpy array of shape (N,3) - state of the network
        state[:,0] = 1 for the susceptible, 0 for the others
        state[:,1] = 1 for the infected, 0 for the others
        state[:,2] = 1 for the recovered, 0 for the others
    beta : prob of infection given contact
    mu : prob of recovery per step
    gamma : prob of S->R transition per step
    T : fraction of the day spent in the system
    
    Return
    ------
    state, recovered (updated)
    """
    N = len(state)
    new_state = np.zeros((N,3))
    
    dprint = print if debug else lambda *args, **kwargs : None
        
    ### S -> I ###
    p_I = beta*np.matmul(A,state[:,1]).T # prob of getting the infection
    p_I = np.array(p_I).reshape(N)
    u = np.random.rand(N) 
    mask_S = (u < p_I*state[:,0]) # apply only to susceptible
    new_state[mask_S,1] = 1
    state[mask_S,0] = 0
    
    dprint("New I: ", new_state[:,1].sum())
    
    ### I -> R ###
    u = np.random.rand(N) 
    mask_I = (u < mu*state[:,1]) # apply only to infected
    new_state[mask_I,2] = 1
    state[mask_I,1] = 0
    
    dprint("New R: ", new_state[:,2].sum())
    
    ### R -> S ###
    u = np.random.rand(N) 
    mask_R = (u < gamma*state[:,2]) # apply only to recovered
    new_state[mask_R,0] = 1
    state[mask_R,2] = 0
    
    dprint("New S: ", new_state[:,0].sum())
    
    state = state + new_state
    
    dprint("Updated S: ", state[:,0].sum())
    dprint("Updated I: ", state[:,1].sum())
    dprint("Updated R: ", state[:,2].sum())
    
    return state

def SIRS_masked_step(A, state, mask, beta, mu, gamma, T=0.5, debug=False):
    """
    SIRS step for a single network. Updated A and state needs to be computed
    before calling this function (take into account mobility + dynamic contacts).
    Works with synchronous update (e.g. new infected cannot recover in this step).
    
    Parameters
    ----------
    A : numpy matrix, adjacency matrix
    state : numpy array of shape (N,3) - state of the network
        state[:,0] = 1 for the susceptible, 0 for the others
        state[:,1] = 1 for the infected, 0 for the others
        state[:,2] = 1 for the recovered, 0 for the others
    mask : True at index i if that node is present in the system at current step
    beta : prob of infection given contact
    mu : prob of recovery per step
    gamma : prob of S->R transition per step
    T : fraction of the day spent in the system
    
    Return
    ------
    state, recovered (updated)
    """
    N = len(state)
    new_state = np.zeros((N,3))
    
    dprint = print if debug else lambda *args, **kwargs : None
        
    ### S -> I ###
    p_I = beta*np.matmul(A,state[:,1]*mask).T
    p_I = np.array(p_I).reshape(N)
    u = np.random.rand(N) 
    mask_S = (u < p_I*state[:,0])
    new_state[mask_S,1] = 1 #new intefected
    state[mask_S,0] = 0 
    
    dprint("New I: ", new_state[:,1].sum())
    
    ### I -> R ###
    u = np.random.rand(N) 
    mask_I = (u < mu*state[:,1]) # apply only to infected 
    new_state[mask_I,2] = 1
    state[mask_I,1] = 0
    
    dprint("New R: ", new_state[:,2].sum())
    
    ### R -> S ###
    u = np.random.rand(N) 
    mask_R = (u < gamma*state[:,2]) # apply only to recovered
    new_state[mask_R,0] = 1
    state[mask_R,2] = 0
    
    dprint("New S: ", new_state[:,0].sum())
    
    state = state + new_state 
    
    dprint("Updated S: ", state[:,0].sum())
    dprint("Updated I: ", state[:,1].sum())
    dprint("Updated R: ", state[:,2].sum())
    
    return state

def attach_travellers_sf(G_sf_stay, new_ids_er, deg_er, N_tot):
    """
    Attach new travellers using preferential attachment and keeping their original degrees.
    
    Parameters
    ----------
    G_sf_stay : Graph instance, graph of the nodes that do not travel
    new_ids_er :  dict, contains the pairs {'new_id_er':old_id}
    deg_er :  numpy array of int, contains the degrees of all the travelling nodes from the ER network
    N_tot : int, number of original nodes + travelling nodes
    
    Returns
    -------
    A_sf_day : numpy matrix, adjacency matrix of G_sf_day 
    """

    edge_list_sf = list(G_sf_stay.edges)
    for i,ID in enumerate(new_ids_er.keys()): 
        k = deg_er[i] 
        indexes = np.random.choice(len(edge_list_sf), size=k, replace=False)
        edges = [(ID,np.random.choice(list(edge_list_sf[j]))) for j in indexes] 
        edge_list_sf += edges
    edge_list_sf = np.array(edge_list_sf) 
    x = edge_list_sf[:,0]
    y = edge_list_sf[:,1]
    A_sf_day = np.zeros((N_tot,N_tot)) 
    A_sf_day[x,y] = 1
    A_sf_day[y,x] = 1     

    return A_sf_day

def attach_travellers_er(G_er_stay, new_ids_sf, deg_sf, N_tot):
    """
    Attach new travellers at random and keeping their original degrees.
    
    Parameters
    ----------
    G_er_stay : Graph instance, graph of the nodes that do not travel
    new_ids_sf :  dict, contains the pairs {'new_id_sf':old_id}
    deg_sf :  numpy array of int, contains the degrees of all the travelling nodes from the SF network
    N_tot : int, number of original nodes + travelling nodes
    
    Returns
    -------
    A_er_day : numpy matrix, adjacency matrix of G_er_day 
    """
    edge_list_er = list(G_er_stay.edges)
    nodes_er = list(G_er_stay.nodes)
    for i,ID in enumerate(new_ids_sf.keys()): 
        k = deg_sf[i] 
        indexes = np.random.choice(nodes_er, size=k, replace=False)
        edges = [(ID,j) for j in indexes] 
        edge_list_er += edges
    edge_list_er = np.array(edge_list_er) 
    x = edge_list_er[:,0]
    y = edge_list_er[:,1]
    A_er_day = np.zeros((N_tot,N_tot)) 
    A_er_day[x,y] = 1
    A_er_day[y,x] = 1     

    return A_er_day

def two_sys_full_SIRS_step(state_sf, state_er, travellers_sf, travellers_er, new_ids_sf, new_ids_er, deg_sf, deg_er, A_sf, A_er, G_sf_stay, G_er_stay, beta, mu, gamma):
    """ 
    Simulate a single step of a SIRS dynamics over 2 coupled network with mobility, 
    taking into account the undelying structure of the networks. 
    
    Parameters
    ----------
    state_sf: numpy array of shape (N,3) - state of the scale free network
        state_sf[:,0] = 1 for the susceptible, 0 for the others
        state_sf[:,1] = 1 for the infected, 0 for the others
        state_sf[:,2] = 1 for the recovered, 0 for the others
    state_er: numpy array of shape (N,3) - state of the Erdosh-Renyi network
    **variables_net_sf (see "prepare_two_sys" function description)
    **variables_net_er (see "prepare_two_sys" function description)
    **infection_params (beta, mu, gamma)
    
    Returns
    -------
    state_sf, state_er (updated)
    """
    N = len(state_sf)
    Nij = len(travellers_sf)
    N_tot = N + Nij
    
    ### day ###
    
    # compute day networks: attach travellers to 
    A_sf_day = attach_travellers_sf(G_sf_stay, new_ids_er, deg_er, N_tot)
    A_er_day = attach_travellers_er(G_er_stay, new_ids_sf, deg_sf, N_tot)
    
    # mobility masks (True if present, False if travelling)
    mob_mask_sf = (~np.isin(np.arange(N_tot), travellers_sf)).astype(int)
    mob_mask_er = (~np.isin(np.arange(N_tot), travellers_er)).astype(int)
    
    # states of the travellers
    state_sf_trav = state_sf[travellers_sf]
    state_er_trav = state_er[travellers_er]

    # stay + travellers of the other system state 
    # also absent travellers are virtually present - that is why we use masks
    state_sf_day = np.concatenate((state_sf, state_er_trav))
    state_er_day = np.concatenate((state_er, state_sf_trav))
    
    # make day SIRS step
    state_sf_day = SIRS_masked_step(A_sf_day, state_sf_day, mob_mask_sf, beta, mu, gamma)
    state_er_day = SIRS_masked_step(A_er_day, state_er_day, mob_mask_er, beta, mu, gamma)
    
    # extract the state of the travellers
    state_sf_trav = state_er_day[N:] 
    state_er_trav = state_sf_day[N:] 

    # overwrite them into the original system 
    state_sf[travellers_sf] = state_sf_trav
    state_er[travellers_er] = state_er_trav
    
    ### night ###
    
    # make SIRS step: i.e. "infection" inside community of residence
    state_sf = SIRS_step(A_sf, state_sf, beta, mu, gamma)
    state_er = SIRS_step(A_er, state_er, beta, mu, gamma)
    
    return state_sf, state_er

def prepare_two_sys(N, I_sf, I_er, p_mob, mean_degree):
    """ 
    Defines two networks, one with a power law distribution (a.k.a. scale-free distribution),
    the other with random connections (binomial or Erdosh-Renyi graph).
    Defines two initial states, one for each network, containing categorical information about the status
    of each node of the network.
    Computes some variables linked to the mobility between the two networks (commuting), 
    used in the SIRS simulation.
    
    Parameters
    ----------
    N : int, number of nodes of each network
    I_sf : int, number of initial infected in the scale-free network
    I_er : int, number of initial infected in the Erdosh-Renyi network
    p_mob : float, probability that each individual has of being a traveller
    mean_degree : (even) int, mean degree of each network
    
    Returns
    -------
    state_sf : numpy array of shape (N,3) - state of the scale free network
        state_sf[:,0] = 1 for the susceptible, 0 for the others
        state_sf[:,1] = 1 for the infected, 0 for the others
        state_sf[:,2] = 1 for the recovered, 0 for the others
        
    state_er : numpy array of shape (N,3) - state of the Erdosh-Renyi network
    
    variables_net_sf : dict, keys = {'travellers_sf', 'new_ids_sf', 'deg_sf', 'A_sf', 'G_sf_stay'}
        travellers_sf : numpy array of int, contains the IDs of the travelling nodes
        new_ids_sf : dict, contains the pairs {'new_id_sf':old_id}
        deg_sf : numpy array of int, contains the degrees of all the travelling nodes
        A_sf : numpy matrix, adjacency matrix of the scale-free network
        G_sf_stay : networkx Graph instance, graph of the nodes that do not travel
        
    variables_net_er : dict, keys = {'travellers_er', 'new_ids_er', 'deg_er', 'A_er', 'G_er_stay'}
    """
    ### Topology ###
    
    p = mean_degree/N # prob of creating an edge
    
    # create networks
    G_sf = nx.barabasi_albert_graph(N,int(mean_degree/2))
    G_er = nx.binomial_graph(N,p)
    
    # get adjacency matrices
    A_sf = nx.to_numpy_matrix(G_sf)
    A_er = nx.to_numpy_matrix(G_er)
    
    ### Initial state ###
    
    state_sf = prepare_init_state(N,I_sf)
    state_er = prepare_init_state(N,I_er)
    
    ### Mobility part ### 
    
    Nij = int(p_mob*N) # number of travellers for each system
    
    # Choose travellers IDs
    travellers_sf = np.random.choice(np.arange(N), size=Nij, replace=False)
    travellers_er = np.random.choice(np.arange(N), size=Nij, replace=False)
    
    # Map the travellers IDs in the other system as N, N+1,...,N+Nij-1
    new_ids_sf = {} 
    for i, ID in enumerate(travellers_sf):
        new_ids_sf[i+N] = ID
    new_ids_er = {} 
    for i, ID in enumerate(travellers_er):
        new_ids_er[i+N] = ID
        
    # Compute the adjacency matrices and the networks of the remainers
    mob_mask_sf = np.isin(np.arange(N), travellers_sf)
    A_sf_stay = np.copy(A_sf)
    #put to 0 elements in the adjacency matrix corresponding to travellers
    A_sf_stay[mob_mask_sf,:] = 0
    A_sf_stay[:,mob_mask_sf] = 0
    G_sf_stay = nx.from_numpy_matrix(A_sf_stay)

    mob_mask_er = np.isin(np.arange(N), travellers_er)
    A_er_stay = np.copy(A_er)
    A_er_stay[mob_mask_er,:] = 0
    A_er_stay[:,mob_mask_er] = 0
    G_er_stay = nx.from_numpy_matrix(A_er_stay)
    
    # Compute the original degrees of the travellers
    deg_sf = [k for n,k in G_sf.degree(travellers_sf)]
    deg_er = [k for n,k in G_er.degree(travellers_er)]
    
    # wrap variables in dictionaries 
    variables_net_sf = {'travellers_sf':travellers_sf, 'new_ids_sf':new_ids_sf, 'deg_sf':deg_sf,
                        'A_sf':A_sf, 'G_sf_stay':G_sf_stay}

    variables_net_er = {'travellers_er':travellers_er, 'new_ids_er':new_ids_er, 'deg_er':deg_er,
                        'A_er':A_er, 'G_er_stay':G_er_stay}
    
    return state_sf, state_er, variables_net_sf, variables_net_er