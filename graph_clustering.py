import ijson
import random
import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib import pylab

def save_graph(graph,n_clusters,cluster_assignments,file_name):
    #initialze Figure
    colormap=['red','blue','green','yellow','cyan','orange','magenta','gray','pink','brown','indigo']
    # print('len of cluster_assgnments array passed to save_graph=',len(cluster_assignments))
    # for j in range(len(cluster_assignments)):
    #     print(j)
    #     print('combined cluster assignment[',j,']=',colormap[cluster_assignments[j]])
    plt.figure(num=None, figsize=(50, 50), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph,pos,node_color=[colormap[cluster_assignments[i]] for i in range(len(cluster_assignments))])
    nx.draw_networkx_edges(graph,pos)
    nx.draw_networkx_labels(graph,pos,font_size=5)
    # nx.draw_networkx(graph,with_labels=True,node_color=[colormap[cluster_assignments[i]] for i in range(len(cluster_assignments))],node_size=20)
    # plt.show()

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    m_xmax = cut * min(xx for xx, yy in pos.values())
    m_ymax = cut * min(yy for xx, yy in pos.values())
    plt.xlim(m_xmax-0.01, xmax+0.01)
    plt.ylim(m_ymax-0.01, ymax+0.01)
    plt.title('Cluster assignments for n_clusters = '+str(n_clusters))
    plt.savefig(file_name,bbox_inches="tight")
    pylab.close()
    del fig

def predict_test_label(cluster_assignments,distance_matrix,train_data_limit,data_len):
    test_cluster_assignments=[]
    for i in range(train_data_limit,data_len):
        distance_array=np.array([distance_matrix[i][neighbour] for neighbour in range(train_data_limit)])
        nearest_neighbour=np.argmin(distance_array)
        if(distance_array[nearest_neighbour]<1e5):
            predicted_cluster=cluster_assignments[nearest_neighbour]
        else:
            predicted_cluster=-1
        test_cluster_assignments.append(predicted_cluster)
    return test_cluster_assignments

dataset_file="dblp.v12.json"
temp_data_cnt=1500
data_explored=0


temporary_dataset=[]
with open(dataset_file, "rb") as f:
    for record in ijson.items(f, "item"):
        if "references" in record:
            node_id=record["id"]
            lst=record["references"]
            temporary_dataset.append((node_id,lst))
        data_explored+=1
        if data_explored > temp_data_cnt:
            break

random.shuffle(temporary_dataset)

adj_list={}
edge_list=[]
fraction=0.8
for record in temporary_dataset[:int(fraction*temp_data_cnt)]:
    adj_list[record[0]]=record[1]
    for neighbour in record[1]:
        temp=[record[0],neighbour]
        edge_list.append(temp)

# create graph
S = nx.Graph()
S.add_edges_from(edge_list)

G=S.subgraph(max(nx.connected_components(S), key=len)).copy()
node_count=len(G.nodes)
edge_count=len(G.edges)
print('Node count =',node_count)
print('Edge count =',edge_count)
print()

nodes_id_order=list(G.nodes)
id_to_order_map={}
for i in range(node_count):
    id_to_order_map[nodes_id_order[i]]=i

id_mapping_file=open(r'graph_info/id_mapping.txt','w')
for node_id in id_to_order_map:
    id_mapping_file.write(str(node_id)+'\t'+str(id_to_order_map[node_id])+'\n')
id_mapping_file.close()

edge_info_file=open(r'graph_info/edge_info.txt','w')
mapped_edge_list=[]
for edge in G.edges:
    u=id_to_order_map[edge[0]]
    v=id_to_order_map[edge[1]]
    temp=[u,v]
    edge_info_file.write(str(u)+'\t'+str(v)+'\n')
    mapped_edge_list.append(temp)
edge_info_file.close()

G_mapped=nx.Graph()
G_mapped.add_nodes_from(np.arange(node_count))
G_mapped.add_edges_from(mapped_edge_list)

path=dict(nx.all_pairs_shortest_path_length(G_mapped,40))
f=open(r'graph_info/sp_info.txt','w')

for i in G_mapped.nodes:
    for j in G_mapped.nodes:
        if j in path[i]:
            f.write(str(i)+'\t'+str(j)+'\t'+str(path[i][j])+'\n')       
f.close()

distance_matrix=np.full((node_count,node_count),1e5)
for i in G_mapped.nodes:
    for j in G_mapped.nodes:
        if j in path[i]:
            distance_matrix[i][j]=path[i][j]

train_data_limit=int(0.7*len(G_mapped.nodes))

silhoutte_score_array=[]

for intended_cluster_cnt in  range(2,11):
    print('Intended cluster count =',intended_cluster_cnt,'...')
    linkage='complete'
    ac=AgglomerativeClustering(n_clusters=intended_cluster_cnt,metric="precomputed",linkage=linkage)
    cluster_assignments=ac.fit_predict(distance_matrix[:train_data_limit,:train_data_limit])

    clusters=[]
    for i in range(intended_cluster_cnt):
        clusters.append([])

    for i in range(train_data_limit):
        clusters[cluster_assignments[i]].append(i)

    output_file=open(r'outputs/output_nclusters'+str(intended_cluster_cnt)+'.txt','w')
    output_file.write('Agglomerative clustering with n_clusters = '+str(intended_cluster_cnt)+' and linkage = "'+linkage+'"\n')
    output_file.write('Graph details: Nodes = '+str(node_count)+', Edges = '+str(edge_count)+'\n\n')
    output_file.write('Results:\n\n')
    output_file.write('Cluster assignments of train data: \n')
    for i in range(intended_cluster_cnt):
        output_file.write('Cluster['+str(i)+']: '+str(clusters[i])+'\n')

    output_file.write('\nCluster assignments of test data: \n')
    test_cluster_assignments=predict_test_label(cluster_assignments,distance_matrix,train_data_limit,node_count)
    for i in range(train_data_limit,node_count):
        output_file.write('Node '+str(i)+': '+str(test_cluster_assignments[i-train_data_limit])+'\n')

    combined_cluster_assignments=list(cluster_assignments)+list(test_cluster_assignments)
    score=silhouette_score(distance_matrix,combined_cluster_assignments)
    output_file.write('Silhoutte score = '+str(score)+'\n')
    output_file.close()
    silhoutte_score_array.append(score)
    save_graph(G_mapped,intended_cluster_cnt,combined_cluster_assignments,r'outputs/clusterviz_ncluster_'+str(intended_cluster_cnt)+'.pdf')

plt.plot(np.arange(2,11),np.array(silhoutte_score_array))
plt.xlabel('n_clusters')
plt.ylabel('Silhoutte score')
plt.title('Silhoutte score vs n_clusters')
plt.savefig(r'outputs/silhoutte_score_vs_nclusters.png')