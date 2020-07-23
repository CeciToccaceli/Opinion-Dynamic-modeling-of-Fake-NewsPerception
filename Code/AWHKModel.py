# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from ndlib.models.DiffusionModel import DiffusionModel
import future.utils
import numpy as np
import random

class AWHkModel(DiffusionModel):
    """
    Model Parameters to be specified via ModelConfig
    :param epsilon: bounded confidence threshold from the HK model, in [0,1]
    Node states are continuous values in [-1,1].
    The initial state is generated randomly uniformly from the domain [-1,1].
    The weight of edges is generated randomly uniformly from the domain [0,1].
    perc_stubborness is in [0,1]
    Stubborn is set to 0 if the node is not stubborn, 1 otherwise.
    option_for_stubbornness is -1 if we compute perc_stubborness only in negatives
    opinion, +1 if we compute perc_stubborness only in positives opinion, 0 if we
    consider union of these two sets (if num_stubborn is compute with uniform distribution over the entire population, 
    option_for_stubborness isn't used)
    """
    def __init__(self, graph):
        """
        Model Constructor
        :param graph: A networkx graph object
        """
        super(self.__class__, self).__init__(graph)
        self.discrete_state = False
        
        self.available_statuses = {
            "Infected": 0
        }
        
        self.parameters = {
            "model": {
                "epsilon": {
                    "descr": "Bounded confidence threshold",
                    "range": [0, 1],
                    "optional": False,
                    "default": 0.1
                },
                "perc_stubborness":{
                    "descr": "Percentage of stubborn agent",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "option_for_stubbornness":{
                    "descr": "Define distribution of stubborns",
                    "range": [-1, 1],
                    "optional": False,
                    "default": 0
                }
            },
            "edges": {
                "weight": {
                    "descr": "Edge weight",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.1
                }
            },
            "nodes":{
                "stubborn":{
                    "descr": "The agent is stubborn or not",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                }
            },
        }
        self.name = "Attractive Weighted HK"
    def set_initial_status(self, configuration=None):
        """
        Override behaviour of methods in class DiffusionModel.
        Overwrites initial status using random real values.
        """
        super(AWHkModel, self).set_initial_status(configuration)
            
        #set node status
        for node in self.status:
            self.status[node] = random.uniform(-1, 1)
        self.initial_status = self.status.copy()
        
    '''
    For each node n, check the status s
    if s isn't included between -1 and 1 excluded, set s to 0.0
    '''
    def clean_initial_status(self, valid_status=None):
        for n, s in future.utils.iteritems(self.status):
            if s > 1 or s < -1:
                self.status[n] = 0.0
                
    def iteration(self, node_status=True):
        
        '''
        Execute a single model iteration

        :return: Iteration_id, Incremental node status (dictionary code -> status)
        '''
        # An iteration changes the opinion of the selected agent 'i' using the following procedure:
        # if i is stubborn then its status doesn't change, else
        # - select a neighbor
        # - if between this pair of agents, there is a smaller distance than epsilon,
        # then there is an attraction between opinions
    

        self.clean_initial_status(None)

        actual_status = {node: nstatus for node, nstatus in future.utils.iteritems(self.status)}
        
        if self.actual_iteration == 0:
            setting = False
            negatives = []
            positives = []
            for node in self.graph.nodes:
                if self.params['nodes']['stubborn'][node] > 0:
                    setting = True
                #if stubborns have intermediate negative opinions    
                '''if actual_status[node]>=-0.6 and actual_status[node]<=-0.4:'''
                if actual_status[node]<=-0.8:
                    if node not in negatives:
                        negatives.append(node)
                #if stubborns have intermediate positive opinions  
                '''if actual_status[node] >= 0.4 and actual_status[node]<=0.6:'''
                if actual_status[node]>=0.8:
                    if node not in positives:
                        positives.append(node)
            
            join_list = negatives + positives
            num_stubborns = 0
            if setting == False:
               #based on the value of option_for_stubbornness, compute num_stubborns or only on negatives, or on positives or on the union of the two
                if self.params['model']['option_for_stubbornness'] == -1 and len(negatives)!= 0:
                    num_stubborns = int(float(len(negatives))*float(self.params['model']['perc_stubborness']))
                elif self.params['model']['option_for_stubbornness'] == 1 and len(positives)!= 0:
                    num_stubborns = int(float(len(positives))*float(self.params['model']['perc_stubborness']))
                elif self.params['model']['option_for_stubbornness'] == 0 and len(join_list) !=0:
                    num_stubborns = int(float(len(join_list))*float(self.params['model']['perc_stubborness']))
                
                count_stub = 0
                
                
                while count_stub < num_stubborns:
                    if self.params['model']['option_for_stubbornness'] == -1:
                        n = random.choice(negatives)
                    elif self.params['model']['option_for_stubbornness'] == 1:
                        n = random.choice(positives)
                    elif self.params['model']['option_for_stubbornness'] == 0:
                        n = random.choice(join_list)

                    if self.params['nodes']['stubborn'][n] == 0:
                        self.params['nodes']['stubborn'][n] = 1
                        count_stub += 1
            
            
                #if num_stubborns is compute with uniform distribution over the entire population (option_for_stubbornness is not used in this case)
                '''
                num_stubborns = 0
                if setting == False:
                    num_stubborns = int(float(self.graph.number_of_nodes())*float(self.params['model']['perc_stubborness']))
                    count_stub = 0

                    while count_stub < num_stubborns:
                        n = list(self.graph.nodes)[np.random.randint(0,self.graph.number_of_nodes())]
                        if self.params['nodes']['stubborn'][n] == 0:
                            self.params['nodes']['stubborn'][n] = 1
                            count_stub += 1
                '''
            
            #in case of LFR: set bridges between communities as stubborn nodes
            ''' 
            if setting == False:
                for b in bridges:
                    if self.params['nodes']['stubborn'][b[0]] == 0:
                        self.params['nodes']['stubborn'][b[0]] = 1
                    if self.params['nodes']['stubborn'][b[1]] == 0:
                        self.params['nodes']['stubborn'][b[1]] = 1                    
            '''
            self.actual_iteration +=1
            delta, node_count, status_delta = self.status_delta(self.status)
            if node_status:
                return {"iteration": 0, "status": self.status.copy(),
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
         
        '''
        - select a random agent n1
        - if it is stubborn:
            its status doesn't change 
        - else:
            - select a random neighbor
            - diff_opinion is compute
            - if diff_opinion < epsilon then :
                attraction: different cases according to the sign of the agents' opinions
        '''
        for i in range(0, self.graph.number_of_nodes()):
            # select a random node
            n1 = list(self.graph.nodes)[np.random.randint(0,self.graph.number_of_nodes())]
            
            #if n1 isn't stubborn
            if self.params['nodes']['stubborn'][n1] == 0:
            
            
                #select neighbors of n1
                neighbours = list(self.graph.neighbors(n1))
                sum_op = 0
                new_op = 0
                
                if len(neighbours) == 0:
                        continue

                        
                #select a neigh for peer interaction
                neigh = random.choice(neighbours)

              
                key = (n1, neigh)
                if key not in list(self.graph.edges()):
                    e = list(key)
                    reverse = [e[1],e[0]]
                    link = tuple(reverse)
                    weight = (self.params['edges']['weight'][link])
                else:
                    weight = (self.params['edges']['weight'][key])
                    
                
                #compute the difference between opinions
                diff_opinion = np.abs((actual_status[n1])-(actual_status[neigh]))
                
                #if diff_opinion < epsilon: attractive interaction
                if diff_opinion <= self.params['model']['epsilon']:
                    sum_op = actual_status[n1]+(actual_status[neigh]*weight)
                
                    #concordant positive signs
                    if (actual_status[n1]>=0 and actual_status[neigh]>=0):
                        if actual_status[n1]> actual_status[neigh]:
                            new_op = actual_status[n1]-((sum_op/2)*(1-actual_status[n1]))
                        elif actual_status[n1]<actual_status[neigh]:
                            new_op = actual_status[n1]+((sum_op/2)*(1-actual_status[n1]))
                        elif actual_status[n1]==actual_status[neigh]:
                            new_op = actual_status[n1]
                    #concordant negative signs
                    if (actual_status[n1]<0 and actual_status[neigh]<0):
                        if actual_status[n1]> actual_status[neigh]:
                            new_op = actual_status[n1]+((sum_op/2)*(1+actual_status[n1]))
                        elif actual_status[n1]<actual_status[neigh]:
                            new_op = actual_status[n1]-((sum_op/2)*(1+actual_status[n1]))
                        elif actual_status[n1]==actual_status[neigh]:
                            new_op = actual_status[n1]
                    #discordant signs
                    if (actual_status[n1]>=0 and actual_status[neigh]<0):
                        if sum_op > 0:
                            new_op = actual_status[n1]-((sum_op/2)*(1-actual_status[n1]))
                        else:
                            new_op = actual_status[n1]+((sum_op/2)*(1-actual_status[n1]))
                    elif (actual_status[n1]<0 and actual_status[neigh]>=0):
                        if sum_op > 0:
                            new_op = actual_status[n1]+((sum_op/2)*(1+actual_status[n1]))
                        else:
                            new_op = actual_status[n1]-((sum_op/2)*(1+actual_status[n1]))
                else: 
                    new_op = actual_status[n1]
                        
            #if n1 is stubborn
            else:
                #opinion doesn't change
                new_op = actual_status[n1]
                
            actual_status[n1] = new_op
        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1
        if node_status:
            return{"iteration":self.actual_iteration-1, "status":delta.copy(), "node_count": node_count.copy(), "status_delta":status_delta.copy()}
        else:
            return{"iteration":self.actual_iteration-1, "status":{}, "node_count":node_count.copy(), "status_delta":status_delta.copy()}
        