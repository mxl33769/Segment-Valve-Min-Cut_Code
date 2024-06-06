# -*- coding: utf-8 -*-
"""
Created on Thu May 13 22:26:58 2021

@author: mxl33
"""



import networkx as nx
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from shapely.geometry import MultiPoint
mpl.rcParams['figure.dpi']=600

#read the data (spreadsheet with valves)
data= pd.read_excel('N_wcu_valves.xlsx', sheet_name=0, index='segment')

#use pandas data frame to store them
grouped = data.groupby('segment')
seg_iso=data.groupby('segment')['valve'].apply(lambda x: list(x)).to_dict()

#generating network structure from the stored data...
valve_dict={}
seg_valve=nx.Graph()
for i in list(seg_iso.keys()):
    for j in list(seg_iso.keys()):
        if j != i:
            for  k in seg_iso[i] :
                if k in seg_iso[j]:
                    seg_valve.add_edge(i,j)
seg_valve.number_of_edges()
seg_valve.number_of_nodes()

#read location data (spreadsheet with pipes)
location_data= pd.read_excel('N_wcu_pipes.xlsx', sheet_name=0, index='segment')
seg_pipe=location_data.groupby('segment')['pipe'].apply(lambda x: list(x)).to_dict()
seg_valve_2=seg_valve
pipe_start_data= pd.read_excel('scarce_wcu_pipe_start.xlsx', sheet_name=0, index='Label')
pipe_start_dict=pipe_start_data.set_index('Label').T.to_dict('list')
pipe_end_data= pd.read_excel('scarce_wcu_pipe_end.xlsx', sheet_name=0, index='Label')
pipe_end_dict=pipe_end_data.set_index('Label').T.to_dict('list')
weight_dict={}
points=[]
pipe_end_data= pd.read_excel('scarce_wcu_pipe_end.xlsx', sheet_name=0, index='Label')
pipe_start_dict=pipe_start_data.set_index('Label').T.to_dict('list')
pipe_end_dict=pipe_end_data.set_index('Label').T.to_dict('list')
xy_data= pd.read_excel('scarce_wcu_junction_xy.xlsx', sheet_name=0, index='Label')
xdata=xy_data['Label'].tolist()
ycoor=xy_data['ycoor'].tolist()
xcoor=xy_data['xcoor'].tolist()
x_dict=dict(zip(xdata, xcoor))
y_dict=dict(zip(xdata, ycoor))
pipes_start=pipe_start_data['Label'].tolist()
pipe_start_data['start']=pipe_start_data['start'].str.strip()
start_nodes=pipe_start_data['start'].tolist()
pipes_stop=pipe_end_data['Label'].tolist()
pipe_end_data['end']=pipe_end_data['end'].str.strip()
stop_nodes=pipe_end_data['end'].tolist()
pipe_start_dict=dict(zip(pipes_start, start_nodes))
pipe_end_dict=dict(zip(pipes_stop, stop_nodes)) 



weight_dict={}
points=[]
for s in list(seg_pipe.keys()):
        
        points=[]
        for t in seg_pipe[s]:
                            start_node = pipe_start_dict[t]
                            end_node = pipe_end_dict[t]
                            start_x=x_dict[start_node]
                            end_x=x_dict[end_node]
                            segment_x=(start_x + end_x)/2
                            start_y=y_dict[start_node]
                            end_y=y_dict[start_node]
                            segment_y=(start_y + end_y)/2
                            points.append((segment_x, segment_y))
        points_up=MultiPoint(points)
        weight_dict[s]=points_up.centroid

pos={}
for key in list(weight_dict.keys()):
    pos[key]= (weight_dict[key].x, weight_dict[key].y)
xy_data= pd.read_excel('scarce_wcu_junction_xy.xlsx', sheet_name=0, index='Label')
xdata=xy_data['Label'].tolist()
ycoor=xy_data['ycoor'].tolist()
xcoor=xy_data['xcoor'].tolist()
x_dict=dict(zip(xdata, xcoor))
y_dict=dict(zip(xdata, ycoor))
pipes_start=pipe_start_data['Label'].tolist()
pipe_start_data['start']=pipe_start_data['start'].str.strip()
start_nodes=pipe_start_data['start'].tolist()
pipes_stop=pipe_end_data['Label'].tolist()
pipe_end_data['end']=pipe_end_data['end'].str.strip()
stop_nodes=pipe_end_data['end'].tolist()
pipe_start_dict=dict(zip(pipes_start, start_nodes))
pipe_end_dict=dict(zip(pipes_stop, stop_nodes))
weight_dict={}
points=[]
weight_dict={}
points=[]
for s in list(seg_pipe.keys()):
        
        points=[]
        for t in seg_pipe[s]:
                            start_node = pipe_start_dict[t]
                            end_node = pipe_end_dict[t]
                            start_x=x_dict[start_node]
                            end_x=x_dict[end_node]
                            segment_x=(start_x + end_x)/2
                            start_y=y_dict[start_node]
                            end_y=y_dict[start_node]
                            segment_y=(start_y + end_y)/2
                            points.append((segment_x, segment_y))
        points_up=MultiPoint(points)
        weight_dict[s]=points_up.centroid

pos={}
for key in list(weight_dict.keys()):
    pos[key]= (weight_dict[key].x, weight_dict[key].y)

nx.set_node_attributes(seg_valve, pos, 'pos')
#nc=nx.draw_networkx(seg_valve, pos= pos, node_size=10, width=1, font_size=2.5)

###########################################################################################

#adding suppersource

from shapely.geometry import MultiPoint
mpl.rcParams['figure.dpi']=600


seg_valve.add_node(158)

seg_valve.add_edge(158, 144)

seg_valve.add_edge(158, 56)

#nx.draw(seg_valve)

##############################################################################
#connectivity

from networkx.algorithms.connectivity import local_node_connectivity
nodes=list(seg_valve.nodes())

connectivity_dict={}
   
for i in nodes:  
   connectivity_dict[i]=local_node_connectivity(seg_valve, 158, i)
   
#######################################################################   
# list of K1, K2

# list of K=1.

d = connectivity_dict

SC_K1 = {k: v for k, v in d.items() if v < 2}

print (SC_K1)
S1= list(SC_K1)
# list of k=2.

d = connectivity_dict

SC_K2 = {k: v for k, v in d.items() if v > 1}

print(SC_K2)   
S2= list(SC_K2)   

#########################################################################
seg_valve.remove_node(158)

#seg_valve.add_edge(158, 144)

#seg_valve.add_edge(158, 56)

#######################################################################

color_map = []
for node in seg_valve:
    if node in S1:
        color_map.append('red')
    if node in S2:
        color_map.append('Yellow')       

nc=nx.draw_networkx(seg_valve, pos= pos,node_color = color_map, node_size=10, width=1, font_size=2.5)

plt.show()




   