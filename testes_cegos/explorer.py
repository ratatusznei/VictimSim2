## EXPLORER AGENT
### @Author: Tacla, UTFPR
### It walks randomly in the environment looking for victims.

import sys
import os
import random
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS

from online_dfs import DFS
from map import Map

import time

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc):
        """ Construtor do agente random on-line
        @param env referencia o ambiente
        @config_file: the absolute path to the explorer's config file
        @param resc referencia o rescuer para poder acorda-lo
        """
        super().__init__(env, config_file)
        self.set_state(VS.ACTIVE)
        
        # Specific initialization for the rescuer
        self.resc = resc           # reference to the rescuer agent   

        # x and y relative to the base
        self.state = (0, 0)

        self.map = Map()
        self.map.insert((0, 0), self.check_for_victim(), 1, self.check_walls_and_lim())

        self.walk_time = 0

        self.cost_to_home = 0
        self.path_to_home = []

        self.exploring = True

        # 7  0  1
        # 6     2
        # 5  4  3
        match self.NAME:
            case 'EXPL_1': exploring_order = [3, 4, 2, 1, 5, 6, 0, 7]
            case 'EXPL_2': exploring_order = [1, 2, 0, 3, 7, 6, 4, 5]
            case 'EXPL_3': exploring_order = [7, 0, 6, 1, 5, 4, 2, 3]
            case 'EXPL_4': exploring_order = [5, 6, 4, 7, 3, 2, 0, 1]
            case _:        print(self.NAME)
        self.dfs = DFS(exploring_order)

    def upload_map_and_stop(self):
        print(f"{self.NAME} No more time to explore... invoking the rescuer")
        self.resc.upload_map(self.map)

    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""
        print(f"\n{self.NAME} deliberate:")

        # No more actions, time almost ended
        print(f'{self.get_rtime()} / {self.walk_time}')

        if self.exploring and self.get_rtime() <= self.cost_to_home + 30.:
            # Time to go home
            # stop exploring and go home
            self.exploring = False

        if self.exploring:
            dx, dy = self.dfs.get_next_walk_action(self.map, self.state)
        else:
            if len(self.path_to_home) > 0:
                dx, dy = self.path_to_home.pop()
            else:
                # time to wake up the rescuer
                # pass the walls and the victims (here, they're empty)
                self.upload_map_and_stop()
                return False
        

        # Moves the body to another position
        start_time = self.get_rtime()
        result = self.walk(dx, dy)
        walk_delta_time = start_time - self.get_rtime()
        self.walk_time += walk_delta_time

        # Test the result of the walk action
        if result == VS.BUMPED:
            walls = 1  # build the map- to do
            print(f"{self.NAME}: wall or grid limit reached")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            print(f"{self.NAME} walk executed, rtime: {self.get_rtime()}")

            self.state = (self.state[0] + dx, self.state[1] + dy)

            if self.state and self.exploring:
                seq = self.check_for_victim()

                if dx == 0 or dy == 0:
                    difficulty = walk_delta_time / self.COST_LINE
                else:
                    difficulty = walk_delta_time / self.COST_DIAG

                self.map.insert(self.state, seq, difficulty, self.check_walls_and_lim())

                self.cost_to_home, self.path_to_home = self.map.get_path(self.state, (0, 0))

                if seq != VS.NO_VICTIM:
                    start_time = self.get_rtime()
                    vs = self.read_vital_signals()
                    read_time = start_time - self.get_rtime()
                    self.walk_time += read_time

                    self.map.insert_victim_signals(self.state, vs)

                    print(f"{self.NAME} Vital signals read, rtime: {self.get_rtime()}")
                    print(f"{self.NAME} Vict: {vs[0]}\n     pSist: {vs[1]:.1f} pDiast: {vs[2]:.1f} qPA: {vs[3]:.1f}")
                    print(f"     pulse: {vs[4]:.1f} frResp: {vs[5]:.1f}")  

        return True

