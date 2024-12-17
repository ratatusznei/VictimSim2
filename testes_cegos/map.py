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

