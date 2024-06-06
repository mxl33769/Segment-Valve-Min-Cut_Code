# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 19:31:51 2021

@author: mxl33
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:44:40 2021

@author: mxl33
"""

import networkx as nx
import pandas as pd
import matplotlib as mpl

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
nc=nx.draw_networkx(seg_valve, pos= pos, node_size=0.5, width=0.02, with_labels=False)

###########################################################################################

#adding suppersource

seg_valve.add_node(158)

seg_valve.add_edge(158, 144)

seg_valve.add_edge(158, 56)


##############################################################################
# random remove
#import random  
#edges=list(seg_valve.edges()) 
#valves=list(seg_valve.edges())
#valves_remove=[]
#valves_remove= random.sample(valves, 22)
#scenarios=[]
#listing_valve=[]
#scenarios.append(valves_remove)
#for listing in scenarios:
#    M_gh=seg_valve
#    for R in range(0, len(listing)):
#        M_gh=nx.contracted_edge(M_gh, listing[R])
#        M_gh.add_node(listing[R][1])
#        listing_valve.append([listing[R]])
##############################################################################  
#nx.draw(M_gh)
##############################################################################
import random  
from itertools import chain

Remove_number= [1] *64

for i in list(Remove_number):
 edges=list(seg_valve.edges()) 
 valves=list(seg_valve.edges())
 valves_remove=[]
 valves_remove= random.sample(valves, i)

 for (u,v) in valves_remove:
  if seg_valve.is_directed():
        in_edges = ((w if w != v else u, u, d)
                    for w, x, d in seg_valve.in_edges(v, data=True))
        out_edges = ((u, w if w != v else u, d)
                     for x, w, d in seg_valve.out_edges(v, data=True))
        new_edges = chain(in_edges, out_edges)
  else:
        new_edges = (
            (u, w if w != v else u, d)
            for x, w, d in seg_valve.edges(v, data=True))
 
  new_edges = list(new_edges)
  seg_valve.remove_node(v)
  M_gh=seg_valve.add_edges_from(new_edges)
  M_gh=seg_valve 

##############################################################################
#connectivity

from networkx.algorithms.connectivity import local_node_connectivity
nodes=list(M_gh.nodes())

connectivity_dict={}
   
for i in nodes:  
   connectivity_dict[i]=local_node_connectivity(M_gh, 158, i)
##############################################################################

#min_cutset

nodes=list(connectivity_dict)

S_cut_K={}

for i in nodes:
    S_cut_K[i]=nx.minimum_node_cut(M_gh, 158, i)

##############################################################################

#Frequency of segment

Frequenc_S_K = list(S_cut_K.values())

print(Frequenc_S_K)

d = Frequenc_S_K
SC_K = []

for i in d:
    if i != []:
        SC_K.append(i)
        
print (SC_K)

##############################################################################


#Frequency of segment 
A = SC_K

import pandas as pd

Result_K = pd.value_counts(A)

s_1 = Result_K 
s_2 = s_1[s_1.values>1]


##############################################################################

#Visualization of results

import matplotlib.pyplot as plt

#设置输出的图片大小
figsize = 11,9
figure, ax = plt.subplots(figsize=figsize)


#设置图例并且设置图例的字体及大小
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}

 
#设置坐标刻度值的大小以及刻度值的字体
plt.yticks(fontsize=15)
plt.xticks(fontsize=18,rotation='Level')
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]


#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}

plt.xlabel('Segment',font2)

plt.ylabel('Number of times segment appears in minimum cuts',font2)
 
data = s_2

data. plot( kind='bar') 

##############################################################################
#Ex2:
##############################################################################
Segments=list(M_gh.nodes())
Segment_remove=[]
Segment_remove1= random.sample(Segments, 5)
SR1=list(Segment_remove1)
M_gh.remove_nodes_from(SR1)
N1=nx.number_connected_components(M_gh)

##############################################################################
Segments=list(M_gh.nodes())
Segment_remove=[]
Segment_remove2= random.sample(Segments, 5)
SR2=list(Segment_remove2)
M_gh.remove_nodes_from(SR2)
N2=nx.number_connected_components(M_gh)

##############################################################################
Segments=list(M_gh.nodes())
Segment_remove=[]
Segment_remove3= random.sample(Segments, 5)
SR3=list(Segment_remove3)
M_gh.remove_nodes_from(SR3)
N3=nx.number_connected_components(M_gh)
##############################################################################
Segments=list(M_gh.nodes())
Segment_remove=[]
Segment_remove4= random.sample(Segments, 5)
SR4=list(Segment_remove4)
M_gh.remove_nodes_from(SR4)
N4=nx.number_connected_components(M_gh)

