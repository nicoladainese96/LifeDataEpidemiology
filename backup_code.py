# load edge list created in Create_Network.ipynb
edge_array_sf = np.load('edges.npy')
edge_list_sf = [tuple(e) for e in edge_array_sf] # change from [[],...,[]] to [(),...,()]
G_SF = nx.Graph(edge_list_sf) # create graph

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
nx.draw(G_sf, node_size = 10, width = 0.3, ax=ax[0])
nx.draw(G_sf1, node_size = 10, width = 0.3, ax=ax[1])