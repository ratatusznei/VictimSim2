from vs.abstract_agent import AbstAgent
from vs.constants import VS

class DFS:
    def __init__(self, exploring_order):
        self.explored: set = {(0, 0)}
        self.exploring_order = exploring_order

        self.untried: dict = {}
        self.walk_queue = []
        self.last_state = None
        self.last_action = None

    def get_next_walk_action(self, map, state):
        if state not in self.untried:
            self.untried[state] = []

            for i in self.exploring_order:
                next_state = (state[0] + AbstAgent.AC_INCR[i][0], state[1] + AbstAgent.AC_INCR[i][1])

                if map.get_neighbors(state)[i] == VS.CLEAR and next_state not in self.explored:
                    self.untried[state].append(AbstAgent.AC_INCR[i])

        if len(self.untried[state]) > 0:
            action = self.untried[state].pop(0)
            next_state = (state[0] + action[0], state[1] + action[1])
        else:
            action = None
            next_state = None

        while next_state in self.explored:
            if len(self.untried[state]) == 0:
                action = None
                break

            action = self.untried[state].pop(0)
            next_state = (state[0] + action[0], state[1] + action[1])

        if action == None:
            if len(self.walk_queue) == 0:
                return (0, 0)

            next_state = self.walk_queue.pop()
            action = (next_state[0] - state[0], next_state[1] - state[1])
        else:
            self.walk_queue.append(state)

        self.explored.add(state)
        self.last_state = state
        self.last_action = action

        return action
