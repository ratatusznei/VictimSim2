import heapq

from vs.abstract_agent import AbstAgent
from vs.constants import VS

class Map:
    def __init__(self):
        self.map: dict = { 
            # (x, y): (vict_id, diff, successors),
        }

        self.victims: dict = {}
 
        self.x_lims = (float('inf'), float('-inf'))
        self.y_lims = (float('inf'), float('-inf'))

    def insert(self, state, victim_id, difficulty, neighbors):
        self.map[state] = (victim_id, difficulty, neighbors)
        self.x_lims = (min(self.x_lims[0], state[0]), max(self.x_lims[1], state[0]))
        self.y_lims = (min(self.y_lims[0], state[1]), max(self.y_lims[1], state[1]))

    def insert_victim_signals(self, state, signals):
        self.victims[state] = signals

    def get_victim_id(self, state):
        return self.map[state][0]

    def get_difficulty(self, state):
        return self.map[state][1]

    def get_neighbors(self, state):
        return self.map[state][2]

    def get_path(self, start_state, end_state):
        # A*
        def h(state):
            dx = abs(end_state[0] - state[0])
            dy = abs(end_state[1] - state[1])
            return dx*dx + dy*dy

        frontier = []
        heapq.heappush(frontier, (0, start_state))

        def get_in_frontier(state):
            for i, (cost, s) in enumerate(frontier):
                if s == state:
                    return i
            None

        explored = set()

        parent = { start_state: None }

        while True:
            if len(frontier) == 0:
                return None

            path_cost, state = heapq.heappop(frontier)

            if state == end_state:
                moves = []
                cost = 0
                while parent[state] != None:
                    d = (state[0] - parent[state][0], state[1] - parent[state][1])
                    moves.append(d)
                    cost += self.get_difficulty(state)
                    state = parent[state]

                return cost, moves

            explored.add(state)

            for i, n in enumerate(self.get_neighbors(state)):
                next_state = (state[0] + AbstAgent.AC_INCR[i][0], state[1] + AbstAgent.AC_INCR[i][1])

                if next_state in self.map: 
                    i_frontier = get_in_frontier(next_state)
                    new_cost = path_cost + (1.5 if i in [1, 3, 5, 7] else 1) + h(next_state)

                    if next_state not in explored and i_frontier == None:
                        parent[next_state] = state
                        heapq.heappush(frontier, (new_cost, next_state))

                    elif j := get_in_frontier(next_state):
                        if new_cost < frontier[j][0]:
                            parent[next_state] = state
                            frontier[j] = (new_cost, next_state)
                            heapq.heapify(frontier)
