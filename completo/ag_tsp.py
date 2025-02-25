import random
import matplotlib.pyplot as plt
import ag_operadores
import numpy as np

class AGTsp():
    TAM_POP = 32              # tamanho da população
    MAX_GERACOES = 8000        # máximo de gerações por execução
    PROB_CROSSOVER = 0.52     # probabilidade de cruzamento entre dois indivíduos
    PROB_MUTACAO = 0.21       # probabilidade de mutação sobre um indivíduo
    PLOTAR = False
    TEMP = 50

    ct_execucoes = 0          # conta as execucoes

    def __init__(self, N_VICTIMS, fitness_function):
        self.m = [list(range(N_VICTIMS)) for _ in range(2*self.TAM_POP)]
        self.fitness = [0] * (2*self.TAM_POP)

        for i in range(self.TAM_POP*2):
            random.shuffle(self.m[i])

        self.melhor_fit = float('-inf')
        self.melhor_caminho = None

        # list(int) -> float
        self.fitness_function = fitness_function

        if AGTsp.PLOTAR:
            plt.title('Melhor individuo x geracao')
            plt.xlabel('Geracao')
            plt.ylabel('Melhor Fitness')
            
    def sort_by_fitness(self):
        sorted_indices = sorted(range(len(self.fitness)), key=lambda i: self.fitness[i], reverse=True)
        fitness_sorted = [self.fitness[i] for i in sorted_indices]
        m_sorted = [self.m[i] for i in sorted_indices]
        self.fitness = fitness_sorted
        self.m = m_sorted


    def executar_ag(self):
        geracao = 0
        sel = [0] * self.TAM_POP

        # PLOT
        x_geracao = []
        y_melhor_fit = []

        while geracao < AGTsp.MAX_GERACOES:
            for i in range(self.TAM_POP):
                self.fitness[i] = self.fitness_function(self.m[i])

            # Cruza selecionados
            sel = ag_operadores.sel_roleta(self.fitness, self.TAM_POP)
            j = 0
            while j < self.TAM_POP:
                a = j + self.TAM_POP
                b = j + self.TAM_POP + 1
                self.m[a] = self.m[sel[j]].copy()
                self.m[b] = self.m[sel[j + 1]].copy()
                
                ag_operadores.cross_mapeamento_parcial(self.m[a], self.m[b], self.PROB_CROSSOVER * self.TEMP)
                j += 2

            # Calcula fitness dos novos individuos
            for i in range(self.TAM_POP, 2 * self.TAM_POP):
                ag_operadores.mutar(self.m[i], self.PROB_MUTACAO * self.TEMP)
                self.fitness[i] = self.fitness_function(self.m[i])

            # Orderna individuos por fitness
            self.sort_by_fitness()

            if self.fitness[0] > self.melhor_fit:
                self.melhor_fit = self.fitness[0]
                self.melhor_caminho = self.m[0]

            if AGTsp.PLOTAR:
                x_geracao.append(geracao)
                y_melhor_fit.append(self.melhor_fit)
                            
            geracao += 1
            alpha = 0.13
            self.TEMP = 1 * alpha + self.TEMP * (1-alpha)

        if AGTsp.PLOTAR:
            plt.title(f'Melhor individuo x geracao (exec {AGTsp.ct_execucoes+1:d})')
            plt.plot(x_geracao, y_melhor_fit, marker='o', color='blue', markersize='2')
            plt.grid('y')
            plt.show()            # mostra a versao final do grafico

if __name__ == '__main__':
    victims1 = [((-23, -13), [151, 14.380375, 10.146992, 4.082034, 167.963676, 19.845524, 2.0, 2]), ((-41, -19), [86, 0.268582, 0.513716, 0.0, 28.011359, 5.788821, 2.0, 2]), ((-39, -20), [94, 15.308895, 10.171164, 4.963855, 148.534924, 16.388311, 2.0, 2]), ((-38, -20), [100, 19.255294, 10.730913, 7.061195, 150.479318, 14.6348, 2.0, 2]), ((-34, -20), [119, 15.860908, 10.985549, 8.614897, 148.312772, 20.286679, 1.0, 1]), ((-61, -16), [9, 2.30047, 4.007773, 0.0, 30.186009, 7.49459, 2.0, 2]), ((-60, -15), [12, 8.079704, 2.868062, -5.800125, 46.269471, 4.71997, 2.0, 2]), ((-50, -22), [43, 16.20292, 11.792697, 8.676628, 153.288098, 22.0, 1.0, 1]), ((-50, -20), [44, 12.146685, 10.796732, 4.169803, 159.157079, 22.0, 2.0, 2]), ((-48, -19), [54, 1.041932, 0.876345, 0.0, 46.55489, 7.344399, 2.6666666666666665, 2]), ((-22, 6), [155, 17.143989, 10.737435, 7.093797, 161.640103, 20.838105, 1.0, 1])]

    victims = [((32, 67), [285, 7.099206, 3.818551, -7.711726, 50.413344, 8.733424, np.float64(2.0), np.int64(2)]), ((32, 82), [286, 18.523225, 11.760269, 8.661353, 156.787196, 22.0, np.float64(1.0), np.int64(1)]), ((26, 83), [332, 3.371858, 2.006981, 0.0, 46.765778, 7.319149, np.float64(2.0), np.int64(2)]), ((25, 84), [254, 3.501087, 4.261551, 0.0, 33.33335, 8.449426, np.float64(2.0), np.int64(2)]), ((26, 82), [262, 18.626362, 12.067454, 8.733333, 157.808113, 20.137731, np.float64(1.0), np.int64(1)]), ((31, 78), [282, 20.326719, 14.295666, 8.733333, 149.077555, 15.014342, np.float64(2.0), np.int64(2)]), ((31, 77), [281, 14.137866, 11.810098, 6.506578, 145.409581, 18.941913, np.float64(2.0), np.int64(2)]), ((31, 75), [280, 16.151345, 12.609846, 8.733333, 150.323062, 20.505236, np.float64(1.0), np.int64(1)]), ((28, 75), [269, 4.473824, 2.937209, 0.0, 40.618586, 3.551972, np.float64(2.0), np.int64(2)]), ((30, 72), [278, 18.589097, 12.306158, 8.733333, 149.552642, 21.782753, np.float64(1.0), np.int64(1)]), ((17, 85), [333, 17.662683, 11.719484, 8.640265, 165.220177, 17.884912, np.float64(2.0), np.int64(2)]), ((29, 70), [273, 11.75717, 4.814015, -0.530207, 67.271681, 11.674063, np.float64(3.0), np.int64(3)]), ((19, 80), [336, 4.791107, 2.343184, 0.0, 40.663907, 5.666904, np.float64(2.0), np.int64(2)]), ((29, 68), [344, 15.833068, 3.812323, -4.409206, 53.998223, 5.028403, np.float64(2.0), np.int64(2)]), ((21, 76), [243, 16.39837, 14.010772, 8.733333, 152.742018, 18.284363, np.float64(2.0), np.int64(2)]), ((17, 80), [232, 10.430767, 4.729841, -0.921991, 48.983628, 4.485933, np.float64(2.5), np.int64(2)]), ((16, 81), [226, 15.781888, 13.196251, 8.726256, 163.318809, 17.909292, np.float64(2.0), np.int64(2)]), ((15, 82), [341, 17.628909, 12.552978, 8.733333, 163.035643, 18.163825, np.float64(2.0), np.int64(2)]), ((11, 85), [338, 11.83974, 8.32684, -0.0, 90.822406, 16.383024, np.float64(4.0), np.int64(4)]), ((26, 70), [261, 15.2597, 13.768992, 8.686053, 168.768032, 22.0, np.float64(1.0), np.int64(1)]), ((28, 68), [339, 13.484362, 11.301346, 5.552696, 172.822807, 16.77788, np.float64(2.0), np.int64(2)]), ((27, 68), [335, 19.682596, 11.713315, 8.636926, 151.971769, 22.0, np.float64(1.0), np.int64(1)]), ((24, 71), [329, 18.421479, 11.169532, 8.692281, 156.994635, 17.246221, np.float64(2.0), np.int64(2)]), ((17, 77), [345, 7.4173, 3.439998, -6.712491, 44.904751, 8.951952, np.float64(2.0), np.int64(2)]), ((22, 72), [342, 14.528396, 11.585341, 7.226374, 139.71638, 21.57935, np.float64(1.0), np.int64(1)]), ((21, 72), [321, 17.141882, 14.617335, 8.733333, 156.52122, 19.473394, np.float64(1.0), np.int64(1)]), ((17, 76), [231, 5.576765, 3.97495, -8.370013, 54.707378, 4.553263, np.float64(2.0), np.int64(2)]), ((14, 80), [217, 7.340551, 3.401844, -6.953653, 58.312159, 2.311087, np.float64(2.0), np.int64(2)]), ((13, 81), [324, 3.480482, 1.478973, 0.0, 40.679962, 6.091615, np.float64(2.0), np.int64(2)]), ((8, 85), [193, 11.607988, 6.879745, -0.0, 84.042535, 15.226936, np.float64(4.0), np.int64(4)]), ((9, 82), [196, 7.793315, 0.813978, -6.148554, 48.165004, 7.45447, np.float64(2.0), np.int64(2)]), ((13, 78), [211, 12.539345, 11.039033, 4.871346, 174.125612, 22.0, np.float64(1.0), np.int64(1)])]

    print(len(victims1))
    print(len(victims))


    def fit(victim_indexes: list[int]):
        v0 = victims[victim_indexes[0]]
        dist = abs(v0[0][0]) + abs(v0[0][1])

        for i in range(len(victim_indexes)-1):
            v0 = victims[victim_indexes[i]]
            v1 = victims[victim_indexes[i+1]]
            dist += abs(v0[0][0] - v1[0][0]) + abs(v0[0][1] - v1[0][1])

        v0 = victims[victim_indexes[-1]]
        dist += abs(v0[0][0]) + abs(v0[0][1])

        return -dist

    AGTsp.PLOTAR = True 
    a = AGTsp(len(victims), fit)
    a.executar_ag()

