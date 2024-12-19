##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim

import os
import random
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod

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
            clusters = self.cluster_victims()

            for i in range(4):    
                self.rescuers[i].map = self.map     # each rescuer have the map
                self.rescuers[i].victims = clusters[i]     # each rescuer have the map

                self.rescuers[i].planner()
                self.rescuers[i].set_state(VS.ACTIVE)

            print(f'{self.NAME} received all maps')
        
    def cluster_victims(self):
        nw_victims = []
        ne_victims = []
        sw_victims = []
        se_victims = []

        midpoint = ((self.map.x_lims[0] + self.map.x_lims[1]) // 2, (self.map.y_lims[0] + self.map.y_lims[1]) // 2)
        for state, victim_signals in self.victims.items():
            if state[0] < midpoint[0]:
                if state[1] < midpoint[1]:
                    nw_victims.append((state, victim_signals))
                else:
                    ne_victims.append((state, victim_signals))
            else:
                if state[1] < midpoint[1]:
                    sw_victims.append((state, victim_signals))
                else:
                    se_victims.append((state, victim_signals))

        return [nw_victims, ne_victims, sw_victims, se_victims]

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

        while len(self.victims) > 0:
            v = pop_closest(plan_state, self.victims)
            _, path_to = self.map.get_path(plan_state, v[0])
            path_to.reverse()
            self.plan += path_to
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

