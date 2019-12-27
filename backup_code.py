# load edge list created in Create_Network.ipynb
edge_array_sf = np.load('edges.npy')
edge_list_sf = [tuple(e) for e in edge_array_sf] # change from [[],...,[]] to [(),...,()]
G_SF = nx.Graph(edge_list_sf) # create graph