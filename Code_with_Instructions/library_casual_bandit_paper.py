import networkx as nx
import itertools as itr
import random
from typing import Union, List
import numpy as np
from graph_utils import fill_vstructures, direct_chordal_graph
from scipy.special import binom
import pyAgrum as gum
from itertools import combinations
import numpy as np
from scipy.stats import bernoulli
import math


def has_directed_path(arcs, start, end):
    graph = {}
    for u, v in arcs:
        if u not in graph:
            graph[u] = []
        graph[u].append(v)

    visited = set()

    def dfs(node):
        if node == end:
            return True
        visited.add(node)
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
        return False

    return dfs(start)



def rSubset(arr, r):
 
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(arr, r))

def shanmugam_random_chordal(nnodes, density):
    while True:
        d = nx.DiGraph()
        d.add_nodes_from(set(range(nnodes)))
        order = list(range(1, nnodes))
        for i in order:
            num_parents_i = max(1, np.random.binomial(i, density))
            parents_i = random.sample(list(range(i)), num_parents_i)
            d.add_edges_from({(p, i) for p in parents_i})
        for i in reversed(order):
            for j, k in itr.combinations(d.predecessors(i), 2):
                d.add_edge(min(j, k), max(j, k))

        perm = np.random.permutation(list(range(nnodes)))
        d = nx.relabel.relabel_nodes(d, dict(enumerate(perm)))

        return d

def adj_list_to_string_with_vertices(adj_list):
    vertices = sorted(set(v for edge in adj_list for v in edge))
    vertices_str = ';'.join(map(str, vertices))

    adj_str = ';'.join([f"{edge[0]}->{edge[1]}" for edge in adj_list])

    return f"{vertices_str};{adj_str}"


def block_sample_intervention(bn,intv1,intv2,samples):
    import numpy as np
    import math
    bn1  = gum.BayesNet(bn)
    import numpy as np
    for j in intv1:
        for i in bn1.parents(j):
            bn1.eraseArc(gum.Arc(i,j))
       
       
        bn1.cpt(j)[:] = [0.5,0.5]

    for j in intv2:
        for i in bn1.parents(j):
            bn1.eraseArc(gum.Arc(i,j))
       
       
        bn1.cpt(j)[:] = [0,1]

           
                                        
            
    df = gum.generateSample(bn1, n=math.ceil(samples), name_out=None, show_progress=False, with_labels=True, random_order=False)
    aa = df[0]
    return aa 

def data_gen(nodes,degree,latent_deg,num_of_graphs):    
    s_obs_L =[]
    s_t_L =[]
    s_pomis_L=[]
    for j in range(0,num_of_graphs,1):
    

    
        a = shanmugam_random_chordal(nodes,degree)
        adjacency_list = list(a.edges)
        latent_edge = [];
        adjacency_list1 = adjacency_list.copy()
        tmp = adj_list_to_string_with_vertices(adjacency_list1)
        bn = gum.fastBN(tmp)
        
        confounded = []
        
        n_list = list(range(0,nodes,1))
        combs = rSubset(n_list, 2)
        
        crr = nodes
        for j in combs:
          if( bernoulli.rvs(latent_deg, size = 1)):
                                               adjacency_list1.append((crr,j[0]))
                                               adjacency_list1.append((crr,j[1]))
                                               latent_edge.append(j) 
                                               crr = crr+1
                                               confounded.append(j)
        #print(confounded)
        delta = 0.99
        d_max = nodes
        alpha = 2*nodes*math.log(2/delta+2)/math.log(nodes)
        d1 = delta/(32*alpha*d_max*nodes*math.log(nodes))
        d2 = d1*nodes
        d3 = d2
        d4 = d3
        gamma = 0.1
        eta = 0.01
        A = 8/gamma**2*math.log(2*nodes*4/d1)
        B = 8/gamma**2*math.log(2*nodes*4/d2)
        
        C = 16/(eta*gamma**2)*math.log(2*nodes**2*4/d3) + 1/(2*eta**2)*math.log(2*nodes**2*4/d4)
        
        
        
        Target = []
        samples = []
        tst_pair = 0
        Total_tests  = 0
        Crr_nodesnew = []
        
        for i in n_list:
        
             if(has_directed_path(adjacency_list, i, nodes-1)):
                                                                   Crr_nodesnew.append(i) 
        
   
        Crr_nodesnew.remove(nodes-1)
        alpha = 2*len(Crr_nodesnew)*math.log(2/delta+2)/math.log(nodes)*degree
  
        s_obs = 8*alpha*len(Crr_nodesnew)*math.log(nodes)*(2*A*nodes+B)
        s_t =s_obs +  8*alpha*len(Crr_nodesnew)*(max(0,C-B))*degree    
        for j in range(0,math.floor(8*alpha*degree*len(Crr_nodesnew)*nodes*math.log(nodes))):
            Target_j =[]  
        
            for i in range(0,nodes):
                   
                        if( bernoulli.rvs(1-1/(2*nodes), size = 1) ):
                             Target_j.append(i)    
            Target.append(Target_j)
            samples.append(0)
            
        MUCTnew =[nodes-1]
        MUCT =[]

        while (MUCTnew != MUCT):
                       MUCT = MUCTnew.copy()
                       Crr_nodes = Crr_nodesnew.copy()
                       for i in MUCT:
                                for j in Crr_nodes:
                                   
                                                      if((i,j) in confounded or (j,i) in confounded  ):
                                                                                                             
                                                                                                           tst_pair = tst_pair +1     
                                                                                                           if(j not in MUCTnew ):
                                                                                                               MUCTnew.append(j)
                                                                                                           if (j in  Crr_nodesnew):
                                                                                                                    Crr_nodesnew.remove(j)
                      
                                                      if((i,j) in adjacency_list):
                                                            
                                                                                                           if(j not in MUCTnew ):
                                                                                                               MUCTnew.append(j)
                                                                                                           if (j in  Crr_nodesnew):
                                                                                                                    Crr_nodesnew.remove(j)   
                       #print(MUCT) 
                       #print(MUCTnew)
                       Crr_nodes = Crr_nodesnew
        
        
        #print(tst_pair)
        s_pomis =s_obs +  tst_pair*(max(0,C-B))
        #print(s_obs/1000000,s_t/1000000,s_pomis/1000000)
       
        s_pomis_L.append(s_pomis)
        s_t_L.append(s_t)
        s_obs_L.append(s_obs)  


    return  s_pomis_L,s_t_L,s_obs_L

def plot(Data_save,color_line,plt,n_range,plt_label):
        dd  = Data_save
        if (plt == 0):
                        import matplotlib.pyplot as plt
                       
        m  = np.mean(dd,axis = 0)
        sd = np.std(dd,axis =0)/(math.sqrt(dd.shape[0]))
        mup = m+sd
        mlp = m-sd
        
       
        color_area = color_line + (1-color_line)*2.3/4 ;
        
        plt.plot(n_range,m,color= color_line,label = plt_label)
        plt.fill_between(n_range,mup,mlp,color=color_area)
        plt.xlabel('Number of Nodes in Graph')
        plt.ylabel('Interventional Samples')
    
        plt.grid(True)
        plt.legend()
        return  plt
    
def comapre_alg4_with_full_disc(n_range,rho,rho_L,num_of_graphs):
    s_p =[]
    s_t =[]
    s_o =[] 

    for i in n_range:
      print(i)  
      a,b,c = data_gen(i,rho,rho_L,num_of_graphs)
      s_p.append(a)
      s_t.append(b) 
      s_o.append(c)
    
    s_t = np.array(s_t)
    s_o = np.array(s_o)
    s_p = np.array(s_p)
    s_p = np.transpose(s_p)
    s_t = np.transpose(s_t)
    s_o = np.transpose(s_o)
    return s_o,s_p,s_t
     


def sample_arm(bn,int,real):
    import numpy as np
    import math
    bn1  = gum.BayesNet(bn)
    import numpy as np
    for j in int:
        if (j == list(bn.nodes())[-1]):
                                        continue
        for i in bn1.parents(j):
            bn1.eraseArc(gum.Arc(i,j))
       
       
       
        
        bn1.cpt(j)[:] = [1-real[int.index(j)],real[int.index(j)]]


           
                                        
            
    df = gum.generateSample(bn1, n=math.ceil(1), name_out=None, show_progress=False, with_labels=True, random_order=False)
    aa = df[0]

    return aa 

def sample_arm_samples(bn,int,real,SAMPLES):
    import numpy as np
    import math
    bn1  = gum.BayesNet(bn)
    import numpy as np
    for j in int:
        if (j == list(bn.nodes())[-1]):
                                        continue
        for i in bn1.parents(j):
            bn1.eraseArc(gum.Arc(i,j))
       
       
       
        
        bn1.cpt(j)[:] = [1-real[int.index(j)],real[int.index(j)]]


           
                                        
            
    df = gum.generateSample(bn1, n=math.ceil(SAMPLES), name_out=None, show_progress=False, with_labels=True, random_order=False)
    aa = df[0]

    return aa 

def Learn_transitve_closure(bn,n_list ,Target_j,reward,delta):
    import math
    Edges = []
    Idata = []
    nodes = len(n_list)
  
    d_max = nodes
    alpha = 2*nodes*math.log(2/delta+2)/math.log(nodes)
    d1 = delta/(32*alpha*d_max*nodes*math.log(nodes))
    d2 = d1*nodes
    d3 = d2
    d4 = d3
        
    epsilon = 0.1
    A = 8/epsilon**2*math.log(2*nodes*4/d1)
    B = 8/epsilon**2*math.log(2*nodes*4/d2)
        
    #print(A,B)
    Idata.append(block_sample_intervention(bn,[],Target_j,B,n_list))
    data = np.array(Idata[-1],dtype=int)
    reward.append(data[:,n_list[-1]])
    tst_list =[]
    for i in n_list:
         if (i in Target_j):
                             continue
         tst_list.append(i)
         Idata.append(block_sample_intervention(bn,[i],Target_j,A,n_list))
         data = np.array(Idata[-1],dtype=int)
         reward.append(data[:,n_list[-1]])
    combs = rSubset(tst_list, 2)
    for pair in  combs:
        i_index = tst_list.index(pair[0])+1
        j_index = tst_list.index(pair[1])+1
        tmp = np.array(Idata[0],dtype=int)
        p_j = np.mean(tmp[:,pair[1]])
        tmp = np.array(Idata[i_index],dtype=int)
        p_ij =np.sum(np.multiply(tmp[:,pair[1]],tmp[:,pair[0]]))/np.sum(tmp[:,pair[0]])
        if(abs(p_ij - p_j) >= 0.05):
                          Edges.append(pair)  
        #print(pair)
        #print( i_index,j_index)
        #print(p_ij,p_j) 
        tmp = np.array(Idata[0],dtype=int)
        p_i = np.mean(tmp[:,pair[0]])
        tmp = np.array(Idata[j_index],dtype=int)
        p_ji =np.sum(np.multiply(tmp[:,pair[0]],tmp[:,pair[1]]))/np.sum(tmp[:,pair[1]])
        if(abs(p_ji - p_i) >= 0.05 and abs(p_ij - p_j) <= abs(p_ji - p_i) ):
               if pair in Edges:
                                    Edges.remove(pair)                 
               Edges.append((pair[1],pair[0])) 
        #print(p_ji,p_i)
        #print('****')

    #print(Edges)
    return Edges,reward



def sample_arm(bn,int,real):
    import numpy as np
    import math
    bn1  = gum.BayesNet(bn)
    import numpy as np
    for j in int:
        if (j == list(bn.nodes())[-1]):
                                        continue
        for i in bn1.parents(j):
            bn1.eraseArc(gum.Arc(i,j))
       
       
       
        
        bn1.cpt(j)[:] = [1-real[int.index(j)],real[int.index(j)]]


           
                                        
            
    df = gum.generateSample(bn1, n=math.ceil(1), name_out=None, show_progress=False, with_labels=True, random_order=False)
    aa = df[0]

    return aa 

def sample_arm_samples(bn,int,real,SAMPLES):
    import numpy as np
    import math
    bn1  = gum.BayesNet(bn)
    import numpy as np
    for j in int:
        if (j == list(bn.nodes())[-1]):
                                        continue
        for i in bn1.parents(j):
            bn1.eraseArc(gum.Arc(i,j))
       
       
       
        
        bn1.cpt(j)[:] = [1-real[int.index(j)],real[int.index(j)]]


           
                                        
            
    df = gum.generateSample(bn1, n=math.ceil(SAMPLES), name_out=None, show_progress=False, with_labels=True, random_order=False)
    aa = df[0]

    return aa 
    
def has_directed_path(arcs, start, end):
    graph = {}
    for u, v in arcs:
        if u not in graph:
            graph[u] = []
        graph[u].append(v)

    visited = set()

    def dfs(node):
        if node == end:
            return True
        visited.add(node)
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
        return False

    return dfs(start)



def rSubset(arr, r):
 
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(arr, r))


def decimal_to_binary_list(decimal_num, num_bits):
    binary_num = bin(decimal_num)[2:]  # Convert decimal to binary string
    binary_num = binary_num.zfill(num_bits)  # Pad with zeros to achieve desired number of bits
    binary_list = [int(bit) for bit in binary_num]  # Convert binary string to list of integers
    return binary_list


def shanmugam_random_chordal(nnodes, density):
    while True:
        d = nx.DiGraph()
        d.add_nodes_from(set(range(nnodes)))
        order = list(range(1, nnodes))
        for i in order:
            num_parents_i = max(1, np.random.binomial(i, density))
            parents_i = random.sample(list(range(i)), num_parents_i)
            d.add_edges_from({(p, i) for p in parents_i})
        for i in reversed(order):
            for j, k in itr.combinations(d.predecessors(i), 2):
                d.add_edge(min(j, k), max(j, k))

        perm = np.random.permutation(list(range(nnodes)))
        d = nx.relabel.relabel_nodes(d, dict(enumerate(perm)))

        return d

def adj_list_to_string_with_vertices(adj_list):
    vertices = sorted(set(v for edge in adj_list for v in edge))
    vertices_str = ';'.join(map(str, vertices))

    adj_str = ';'.join([f"{edge[0]}->{edge[1]}" for edge in adj_list])

    return f"{vertices_str};{adj_str}"


def power_set(s):
    s_list = list(s)
    n = len(s_list)
    result = []
    for i in range(2**n):
        subset = []
        for j in range(n):
            if (i >> j) & 1:
                subset.append(s_list[j])
        result.append(subset)
    return result
   




def block_sample_intervention(bn,intv1,intv2,samples,n_list):
    import numpy as np
    import math
    bn1  = gum.BayesNet(bn)
    import numpy as np
    for j in intv1:
        if (j == n_list[-1]):
                                        continue
        for i in bn1.parents(j):
            bn1.eraseArc(gum.Arc(i,j))
       
       
        bn1.cpt(j)[:] = [0.5,0.5]

    for j in intv2:
        if (j == n_list[-1]):
                                        continue
        for i in bn1.parents(j):
            bn1.eraseArc(gum.Arc(i,j))
       
       
        bn1.cpt(j)[:] = [0,1]

           
                                        
            
    df = gum.generateSample(bn1, n=math.ceil(samples), name_out=None, show_progress=False, with_labels=True, random_order=False)
    aa = df[0]
    return aa 


import networkx as nx
import matplotlib.pyplot as plt

def run_full_alg_with_UCB(nodes,degree, latent_deg , num_of_graphs,Max_samples):

        fd_list =[]
        pd_list=[]
        opt_list=[]
        
        
        for iii in range(0,num_of_graphs,1):     
                a = shanmugam_random_chordal(nodes,degree)
                adjacency_list = list(a.edges)
                latent_edge = [];
                adjacency_list1 = adjacency_list.copy()
                tmp = adj_list_to_string_with_vertices(adjacency_list1)
                bn = gum.fastBN(tmp)
                
                      
                confounded = []
                        
                n_list = list(range(0,nodes,1))
                combs = rSubset(n_list, 2)
                        
                crr = nodes
                for j in combs:
                    if( bernoulli.rvs(latent_deg, size = 1)):
                                                               adjacency_list1.append((crr,j[0]))
                                                               adjacency_list1.append((crr,j[1]))
                                                               latent_edge.append(j) 
                                                               crr = crr+1
                                                               confounded.append(j)
                        
        
                tmp = adj_list_to_string_with_vertices(adjacency_list1)
                bn = gum.fastBN(tmp) 
                   
                delta = 0.99
                d_max = nodes
                alpha = 2*nodes*math.log(2/delta+2)/math.log(nodes)
                d1 = delta/(32*alpha*d_max*nodes*math.log(nodes))
                d2 = d1*nodes
                d3 = d2
                d4 = d3
                gamma = 0.1
                eta = 0.01
                C = 16/(eta*gamma)*math.log(2*nodes**2*4/d3) + 1/(2*eta)*math.log(2*nodes**2*4/d4)        
                epsilon = 0.1
                Acc_edges =[]
                ACC_bi_edge =[]
                A = 8/epsilon**2*math.log(2*nodes*4/d1)
                B = 8/epsilon**2*math.log(2*nodes*4/d2)
                samples = 0
                Target = []
                reward = [];
                for j in range(0,math.floor(8*alpha*len(n_list)*math.log(nodes))):
                    Target_j =[]  
                
                    for i in range(0,nodes):
                           
                                if( bernoulli.rvs(1-1/(2*nodes*0.7), size = 1) ):
                                     Target_j.append(i)    
                    #print(n_list,Target_j) 
                 
                    Target.append(Target_j)   
                    TC,reward = Learn_transitve_closure(bn,n_list ,Target_j,reward,delta)                               
                    #print(TC)
                    tmp1 =[]
                    for e in   TC:
                         
                          tmp1 = TC.remove(e)
                         
                          if(tmp1 is not None):
                              if( has_directed_path(tmp1, e[0], e[1])):
                                                                     continue
                          if e not in  Acc_edges and (e[1],e[0]) not in  Acc_edges :
                                             Acc_edges.append(e)   
                
                
                reward_p = reward.copy()
                
                Crr_nodesnew = []
                        
                for i in n_list:
                        
                             if(has_directed_path(adjacency_list, i, nodes-1)):
                                                                                   Crr_nodesnew.append(i) 
                
                
                if(1):
                    dag = bn.dag()
                    for i in Crr_nodesnew:
                         for j in  Crr_nodesnew:
                           
                    
                           pair = (i,j) 
                           P = (i,j)  
                           if (dag.hasDirectedPath(P[1],P[0])):
                                                                  pair = (P[1],P[0])
                           target = ((bn.parents(pair[0]) |  bn.parents(pair[1])) - set({pair[0]}))
                           target = target.intersection(set(n_list)) 
                           #print(target)
                           tmp = np.array( block_sample_intervention(bn,[],target,C,n_list),dtype=int) 
                          
                           reward.append(tmp[: ,n_list[-1]])
                           p_j = np.sum(np.multiply(tmp[:,pair[1]],tmp[:,pair[0]]))/np.sum(tmp[:,pair[0]])
                           a= np.sum(tmp[:,pair[0]])
                           target.add(pair[0]) 
                           tmp =  np.array( block_sample_intervention(bn,[],target,A,n_list),dtype=int) 
                           reward.append(tmp[:,n_list[-1]])
                           p_ij =np.sum(np.multiply(tmp[:,pair[1]],tmp[:,pair[0]]))/np.sum(tmp[:,pair[0]])
                           b= np.sum(tmp[:,pair[0]]) 
                           if(abs(p_ij - p_j) >= 0):
                                                  ACC_bi_edge.append(P) 
                
                
                
                
                
                        
                Crr_nodesnew.remove(nodes-1) 
                        
                      
                MUCTnew =[nodes-1]
                MUCT =[]
                        
                while (MUCTnew != MUCT):
                                    MUCT = MUCTnew.copy()
                                    Crr_nodes = Crr_nodesnew.copy()
                                    for i in MUCT:
                                                for j in Crr_nodes:
                                                   
                                                                      if((i,j) in confounded or (j,i) in confounded  ): 
                                                                                                                          pair = (i,j) 
                                                                                                                          if (dag.hasDirectedPath(P[1],P[0])):
                                                                                                                                        pair = (P[1],P[0])
                                                                                                                          target = ((bn.parents(pair[0]) |  bn.parents(pair[1])) - set({pair[0]}))
                                                                                                                          target = target.intersection(set(n_list)) 
                      
                                                                                                                          tmp = np.array( block_sample_intervention(bn,[],target,C,n_list),dtype=int) 
                      
                                                                                                                          reward_p.append(tmp[: ,n_list[-1]])  
                                                                                                                          
                                                                                                                             
                                                                                                                          
                                                                                                                          if(j not in MUCTnew ):
                                                                                                                               MUCTnew.append(j)
                                                                                                                          if (j in  Crr_nodesnew):
                                                                                                                                    Crr_nodesnew.remove(j)
                                      
                                                                      if((i,j) in adjacency_list):
                                                                            
                                                                                                                           if(j not in MUCTnew ):
                                                                                                                               MUCTnew.append(j)
                                                                                                                           if (j in  Crr_nodesnew):
                                                                                                                                    Crr_nodesnew.remove(j)   
                                       #print(MUCT) 
                                       #print(MUCTnew)
                                    Crr_nodes = Crr_nodesnew
                        
                #print(MUCT)
                
                tmpp = set(MUCT)
                tmpp.remove(nodes-1)
                PS = power_set(tmpp)
                POMIS = []
                for iii in PS:
                                    #print('**',iii)
                                    Crr_nodesnew = []
                                    pomistmp=set([])
                                    adjacency_listtmp=[]
                                    confoundedtmp =[]
                                    for i in adjacency_list:
                                         if(i[1] not in iii ):
                                                                 adjacency_listtmp.append(i)
                                    #print(adjacency_listtmp)
                                    for i in confounded:
                                         if(i[0] not in iii and i[1] not in iii ):
                                                              confoundedtmp.append(i) 
                                    #print(confoundedtmp) 
                                    for i in n_list:
                        
                                          if(has_directed_path(adjacency_listtmp, i, nodes-1)):
                                                                                   Crr_nodesnew.append(i) 
                        
                                    Crr_nodesnew.remove(nodes-1) 
                        
                      
                                    MUCTnew =[nodes-1]
                                    MUCT =[]
                        
                                    while (MUCTnew != MUCT):
                                       MUCT = MUCTnew.copy()
                                       Crr_nodes = Crr_nodesnew.copy()
                                       for i in MUCT:
                                                for j in Crr_nodes:
                                                                      if((i,j) in confoundedtmp or (j,i) in confoundedtmp or (i,j) in adjacency_listtmp):
                                                                          
                                                                                                                           if(j not in MUCTnew ):
                                                                                                                               MUCTnew.append(j)
                                                                                                                           if (j in  Crr_nodesnew):
                                                                                                                                    Crr_nodesnew.remove(j)
                                      
                                       Crr_nodes = Crr_nodesnew
                                    #print("**" ,MUCT)    
                                    for j in MUCT:
                                         pomistmp =     pomistmp | bn.parents(j)
                                    pomistmp = pomistmp - set(MUCT)
                                    if(pomistmp not in POMIS ):
                                        POMIS.append(pomistmp) 
                
                #print(POMIS)     
                
                Action_set = POMIS
                Actionss =[]
                Realizations =[]
                for s in Action_set:
                    for i in range(2**len(s)):
                         Actionss.append(list(s))
                         Realizations.append(decimal_to_binary_list(i,len(s)))
                total_arms = len(Actionss)
                mhat = np.zeros(total_arms)
                Ntimes = np.ones(total_arms)
          
                choice = []
                ucb = mhat +np.sqrt(2/Ntimes)
                for j in range(0,600000,1):
                      #print(j)
                      
                      crract = np.argmax(ucb)
                      sm =sample_arm(bn,Actionss[crract],Realizations[crract])
                      y = np.array(sm,dtype=int)
                      y = y[:,n_list[-1]][0]
                      reward.append([y])
                      reward_p.append([y])
                      choice.append(crract)
                      #print(crract)
                      mhat[crract] = (Ntimes[crract]*mhat[crract] + y)/(Ntimes[crract]+1)
                      Ntimes[crract] =Ntimes[crract]+1
                      ucb[crract] = mhat[crract] +np.sqrt(2/Ntimes[crract])
                    
                list_of_arrays = reward
                optim_rew = np.max(mhat)
                # Combine all elements into one list
                fd = [item for arr in list_of_arrays for item in arr]
                print(len(fd))
                if(len(fd) <= Max_samples ):
                    sm =sample_arm_samples(bn,Actionss[crract],Realizations[crract],Max_samples -len(fd))
        
                    y = np.array(sm,dtype=int)
                    y = y[:,n_list[-1]]
        
                      
                    reward.append(list(y))   
                list_of_arrays = reward 
                fd = [item for arr in list_of_arrays for item in arr]
        
        
                list_of_arrays = reward_p
                
                # Combine all elements into one list
                pd = [item for arr in list_of_arrays for item in arr]
                print(len(pd))
                if(len(pd) <= Max_samples ):
                    sm =sample_arm_samples(bn,Actionss[crract],Realizations[crract],Max_samples -len(pd))
        
                    y = np.array(sm,dtype=int)
                    y = y[:,n_list[-1]]
        
                      
                    reward_p.append(list(y))   
                
                list_of_arrays = reward_p
                
                # Combine all elements into one list
                pd = [item for arr in list_of_arrays for item in arr]
                fd_a = np.array(fd)
                pd_a = np.array(pd)
        
          
        
              
                fd_list.append(list(fd_a))
                pd_list.append(list(pd_a))
                opt_list.append(optim_rew)
                print('*')
        return fd_list, pd_list ,opt_list     


def plot_reg(Data_save,color_line,plt,T_range,plt_label):
        import numpy as np
        T_range = T_range/100 
        T_range = np.array(range(int(T_range)))
        dd  = Data_save
        if (plt == 0):
                        import matplotlib.pyplot as plt
                       
        m  = np.mean(dd,axis = 0)
        sd = np.std(dd,axis =0)/(math.sqrt(dd.shape[0]))
        mup = m+sd
        mlp = m-sd
        
       
        color_area = color_line + (1-color_line)*2.3/4 ;
        
        plt.plot(T_range,m,color= color_line,label = plt_label)
        plt.fill_between(T_range,mup,mlp,color=color_area)
        plt.xlabel('Number of Nodes in Graph')
        plt.ylabel('Interventional Samples')
    
        plt.grid(True)
        plt.legend()
        return  plt



def comapare_reg_fulldisc_vs_alg4(nodes,degree, latent_deg , num_of_graphs,Max_samples):

    fd_list, pd_list,opt_list= run_full_alg_with_UCB(nodes,degree, latent_deg , num_of_graphs,Max_samples)
    fd_a = np.array(fd_list[0:num_of_graphs-1])
    pd_a = np.array(pd_list[0:num_of_graphs-1])
    opt_a = np.array(opt_list[0:num_of_graphs-1])
    fd_a = np.cumsum(opt_a[...,np.newaxis]-fd_a,axis=1)
    fd_a = fd_a[:,1:Max_samples:100]


    pd_a = np.cumsum(opt_a[...,np.newaxis]-pd_a,axis=1)
    pd_a = pd_a[:,1:Max_samples:100]
    return pd_a,fd_a