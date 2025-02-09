##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim

import os, random, csv
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod

import numpy as np

from sklearn.cluster import KMeans

from map import Map

## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, master=False):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.plan = []              # a list of planned actions
                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

        self.maps_received = 0

        self.map = Map()
        self.victims: dict = {}

        self.clusters = []
        self.sequences = []

        if master: 
            # Instantiate the other rescuers and assign the clusters to them
            self.rescuers = [self, None, None, None]

            for i in range(1, 4):    
                #print(f"{self.NAME} instantianting rescuer {i+1}, {self.get_env()}")
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                # each rescuer receives one cluster of victims
                self.rescuers[i] = Rescuer(self.get_env(), config_file) 
    
    def update_map(self, map, victims):
        """ The explorer sends the map containing the walls and
        victims' location. The rescuer becomes ACTIVE. From now,
        the deliberate method is called by the environment"""
        self.maps_received += 1
        self.map.update_map(map)
        self.victims.update(victims)


        if self.maps_received < 4:
            print(f'{self.NAME} received {self.maps_received} maps')

        else:
            self.predict_severity_and_class()
            clusters = self.cluster_victims()

            for i, cluster in enumerate(clusters):
                self.save_cluster_csv(cluster, i+1)    # file names start at 1

            for i in range(4):    
                self.rescuers[i].map = self.map     # each rescuer have the map

                self.rescuers[i].victims = clusters[i]

                self.rescuers[i].clusters = [clusters[i]]
                self.rescuers[i].planner()

                self.save_sequence_csv(self.rescuers[i].sequences, i) 

                self.rescuers[i].set_state(VS.ACTIVE)

            print(f'{self.NAME} received all maps')
        
    def save_cluster_csv(self, cluster, cluster_id):
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vs in cluster:
                x, y = vs[0]     # x,y coordinates
                vs = vs[1]
                writer.writerow([vs[0], x, y, vs[6], vs[7]])

    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for values in sequence:
                print(values)
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([vs[0], x, y, vs[6], vs[7]])

    def cluster_victims(self):
        n_victims = len(self.victims)
        n_clusters = 4 # int(n_victims**0.5);

        kmeans = KMeans(n_clusters, random_state=1)

        coordinates = np.array(list(self.victims.keys()))
        kmeans.fit(coordinates)

        clusters = [ [] for i in range(n_clusters) ]

        for i, cluster_i in enumerate(kmeans.labels_):
            clusters[cluster_i].append((list(self.victims.keys())[i], self.victims[list(self.victims.keys())[i]]))

        return clusters

    def predict_severity_and_class(self):
        """ @TODO to be replaced by a classifier and a regressor to calculate the class of severity and the severity values.
            This method should add the vital signals(vs) of the self.victims dictionary with these two values.

            This implementation assigns random values to both, severity value and class"""

        data_treino = "../datasets/data_4000v/env_vital_signals.txt"
        data_validacao = "../datasets/data_800v/env_vital_signals.txt"
        df_treino = pd.read_csv(data_treino)
        X_treino = df_treino.drop(df_treino.columns[[0, 1, 2, 6, 7]], axis=1)
        value_treino = df_treino[df_treino.columns[[6]]]
        class_treino = df_treino[df_treino.columns[[7]]]
        df_validacao = pd.read_csv(data_validacao)
        value_validacao = df_validacao.columns[[6]]
        class_validacao = df_validacao.columns[[7]]
        X_validacao = df_validacao.drop(df_validacao.columns[[0, 1, 2, 6, 7]], axis=1)

        # vs é [id, pSist, pDiast, qPA, pulso, freq_resp, gravidade value, gravidade class]
        # usar qPA, pulso e freq_resp APENAS
        def decision_tree_classifier(vs): 
            return random.randint(1, 4)

        def decision_tree_regressor(vs):
            return random.uniform(0.1, 99.9)


        for state, vs in self.victims.items():
            severity_value = decision_tree_regressor(vs)
            severity_class = decision_tree_classifier(vs)
            vs.extend([severity_value, severity_class])  # append to the list of vital signals; values is a pair( (x,y), [<vital signals list>] )

    def planner(self):
        """ A private method that calculates the walk actions to rescue the
        victims. Further actions may be necessary and should be added in the
        deliberata method"""

        def pop_closest(state, victims):
            ci = None
            cd = float('inf')
            for i, v in enumerate(victims):
                dx = v[0][0] - state[0]
                dy = v[0][1] - state[1]
                d = dx*dx + dy*dy 
                if d < cd:
                    cd = d
                    ci = i
            
            return victims.pop(ci)

        # This is a off-line trajectory plan, each element of the list is
        # a pair dx, dy that do the agent walk in the x-axis and/or y-axis
        self.plan = []
        plan_state = (0, 0)

        self.sequences = []

        while len(self.victims) > 0:
            v = pop_closest(plan_state, self.victims)
            _, path_to = self.map.get_path(plan_state, v[0])
            path_to.reverse()
            self.plan += path_to

            self.sequences.append(v)

            plan_state = v[0]

        _, path_back = self.map.get_path(plan_state, (0, 0))
        path_back.reverse()
        self.plan += path_back
        
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)

        # Walk - just one step per deliberation
        result = self.walk(dx, dy)

        # Rescue the victim at the current position
        if result == VS.EXECUTED:
            # check if there is a victim at the current position
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                res = self.first_aid() # True when rescued

        print(f"{self.NAME} remaining time: {self.get_rtime()}")

        return True

