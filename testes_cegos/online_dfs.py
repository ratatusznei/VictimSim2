from vs.abstract_agent import AbstAgent
from vs.constants import VS

class DFS:
    def __init__(self, exploring_order):
        self.explored: set = {(0, 0)}
        self.exploring_order = exploring_order

        self.untried: dict = {}
        self.result: dict = {} 
        self.unbacktracked: dict = {}
        self.last_state = None
        self.last_action = None

    def get_next_walk_action(self, map, state):
        if state not in self.untried:
            self.untried[state] = []

            for i in self.exploring_order:
                next_state = (state[0] + AbstAgent.AC_INCR[i][0], state[1] + AbstAgent.AC_INCR[i][1])

                if map.get_neighbors(state)[i] == VS.CLEAR and next_state not in self.explored:
                    self.untried[state].append(AbstAgent.AC_INCR[i])

        if state is not None:
            if (self.last_state, self.last_action) not in self.result:
                self.result[(self.last_state, self.last_action)] = state

                if state not in self.unbacktracked:
                    self.unbacktracked[state] = []
                
                self.unbacktracked[state].append(self.last_state)

        if len(self.untried[state]) == 0:
            if len(self.unbacktracked[state]) == 0:
                return (0, 0)
            else:
                next_state = self.unbacktracked[state].pop(0)
                action = (next_state[0] - state[0], next_state[1] - state[1])
        else:
            action = self.untried[state].pop(0)


        self.explored.add(state)
        self.last_state = state
        self.last_action = action

        return action
