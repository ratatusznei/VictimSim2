import random
import matplotlib.pyplot as plt
import ag_operadores

class AGTsp():
    TAM_POP = 64              # tamanho da população
    MAX_GERACOES = 200        # máximo de gerações por execução
    PROB_CROSSOVER = 0.6     # probabilidade de cruzamento entre dois indivíduos
    PROB_MUTACAO = 0.5       # probabilidade de mutação sobre um indivíduo
    TEMP = 1.
    PLOTAR = False

    ct_execucoes = 0          # conta as execucoes

    def __init__(self, N_VICTIMS, fitness_function):
        self.m = [list(range(N_VICTIMS)) for _ in range(2*self.TAM_POP)]
        self.fitness = [0] * (2*self.TAM_POP)

        for i in range(self.TAM_POP):
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
            self.TEMP -= 1.0/self.MAX_GERACOES

        if AGTsp.PLOTAR:
            plt.title(f'Melhor individuo x geracao (exec {AGTsp.ct_execucoes+1:d})')
            plt.plot(x_geracao, y_melhor_fit, marker='o', color='blue', markersize='2')
            plt.grid('y')
            plt.show()            # mostra a versao final do grafico

if __name__ == '__main__':
    victims = [((-23, -13), [151, 14.380375, 10.146992, 4.082034, 167.963676, 19.845524, 2.0, 2]), ((-41, -19), [86, 0.268582, 0.513716, 0.0, 28.011359, 5.788821, 2.0, 2]), ((-39, -20), [94, 15.308895, 10.171164, 4.963855, 148.534924, 16.388311, 2.0, 2]), ((-38, -20), [100, 19.255294, 10.730913, 7.061195, 150.479318, 14.6348, 2.0, 2]), ((-34, -20), [119, 15.860908, 10.985549, 8.614897, 148.312772, 20.286679, 1.0, 1]), ((-61, -16), [9, 2.30047, 4.007773, 0.0, 30.186009, 7.49459, 2.0, 2]), ((-60, -15), [12, 8.079704, 2.868062, -5.800125, 46.269471, 4.71997, 2.0, 2]), ((-50, -22), [43, 16.20292, 11.792697, 8.676628, 153.288098, 22.0, 1.0, 1]), ((-50, -20), [44, 12.146685, 10.796732, 4.169803, 159.157079, 22.0, 2.0, 2]), ((-48, -19), [54, 1.041932, 0.876345, 0.0, 46.55489, 7.344399, 2.6666666666666665, 2]), ((-22, 6), [155, 17.143989, 10.737435, 7.093797, 161.640103, 20.838105, 1.0, 1])]

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

