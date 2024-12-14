## EXPLORER AGENT
### @Author: Tacla, UTFPR
### It walks randomly in the environment looking for victims.

import sys
import os
import random
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS

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
        self.x = 0
        self.y = 0

        self.map: dict = { 
            # (x, y): (vict_id, diff, successors),
            (0, 0): (self.check_for_victim(), 1, self.check_walls_and_lim()) 
        }
        self.victims: dict = {}
        self.explored: set = {(0, 0)}

        self.walk_time = 0
        self.walk_stack = []

        # DFS
        self.untried: dict = {}
        self.result: dict = {} 
        self.unbacktracked: dict = {}
        self.last_state = None
        self.last_action = None

    def upload_map_and_stop(self):
        print(f"{self.NAME} No more time to explore... invoking the rescuer")
        self.resc.upload_map(self.map, self.victims)

    def return_home(self):
        return self.walk_stack.pop()

    def random_walk(self):
        return (random.randint(-1, 1), random.randint(-1, 1))

    def walk_clock_wise(self):
        state = (self.x, self.y)
        match self.NAME:
            case 'EXPL_1': bias = 0
            case 'EXPL_2': bias = 2
            case 'EXPL_3': bias = 4
            case 'EXPL_4': bias = 6
            case _:        print(self.NAME)

        for i in range(len(self.map[state][2])):
            if self.map[state][2][(i+bias)%8] == VS.CLEAR:
                dx, dy = self.AC_INCR[(i+bias)%8]
                if (self.x+dx, self.y+dy) not in self.explored:
                    return (dx, dy)
        return (0, 0)

    def walk_online_dfs(self):
        state = (self.x, self.y)
        if state not in self.untried:
            self.untried[state] = []

            # 7  0  1
            # 6     2
            # 5  4  3
            match self.NAME:
                case 'EXPL_1': exploring_order = [3, 4, 2, 1, 5, 6, 0, 7]
                case 'EXPL_2': exploring_order = [1, 2, 0, 3, 7, 6, 4, 5]
                case 'EXPL_3': exploring_order = [7, 0, 6, 1, 5, 4, 2, 3]
                case 'EXPL_4': exploring_order = [5, 6, 4, 7, 3, 2, 0, 1]
                case _:        print(self.NAME)

            for i in exploring_order:
                next_state = (self.x + self.AC_INCR[i][0], self.y + self.AC_INCR[i][1])

                if self.map[state][2][i] == VS.CLEAR and next_state not in self.explored:
                    self.untried[state].append(self.AC_INCR[i])

        if state is not None:
            self.result[(self.last_state, self.last_action)] = state
            if state not in self.unbacktracked:
                self.unbacktracked[state] = []
            
            if self.last_state is not None:
                self.unbacktracked[state].append(self.last_state)

        if len(self.untried[state]) == 0:
            if len(self.unbacktracked[state]) == 0:
                return (0, 0)
            else:
                next_state = self.unbacktracked[state].pop(0)
                action = (next_state[0] - state[0], next_state[1] - state[1])
        else:
            action = self.untried[state].pop(0)


        self.last_state = state
        self.last_action = action

        return action

    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""
        print(f"\n{self.NAME} deliberate:")

        # No more actions, time almost ended
        print(f'{self.get_rtime()} / {self.walk_time}')
        if self.get_rtime() <= self.walk_time + 3:
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            if len(self.walk_stack) > 0:
                dx, dy = self.return_home()
            else:
                self.upload_map_and_stop()
                return False
        else:
            dx, dy = self.walk_online_dfs()
        

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

            self.x += dx
            self.y += dy

            if (self.x, self.y) not in self.explored:
                self.explored.add((self.x, self.y))
                self.walk_stack.append((-dx, -dy))

                seq = self.check_for_victim()

                difficulty = walk_delta_time / self.COST_LINE if dx == 0 or dy == 0 else walk_delta_time / self.COST_DIAG
                self.map[(self.x, self.y)] = (seq, difficulty, self.check_walls_and_lim())

                if seq != VS.NO_VICTIM:
                    start_time = self.get_rtime()
                    vs = self.read_vital_signals()
                    read_time = start_time - self.get_rtime()
                    self.walk_time += read_time

                    self.victims[(self.x, self.y)] = vs

                    print(f"{self.NAME} Vital signals read, rtime: {self.get_rtime()}")
                    print(f"{self.NAME} Vict: {vs[0]}\n     pSist: {vs[1]:.1f} pDiast: {vs[2]:.1f} qPA: {vs[3]:.1f}")
                    print(f"     pulse: {vs[4]:.1f} frResp: {vs[5]:.1f}")  

        return True

