## Algoritmos genéticos
## Operadores de seleção, crossover e mutação para codificações binárias de indivíduos.
## 
##  Códificação posicional
## 
import random
from typing import List

def sel_roleta(fitness: List[float], tam_pop) -> List[int]:
    selecao = []

    min_fit = min(fitness)
    fitness_plus = [f+min_fit for f in fitness]

    fitness_rel = [f / sum(fitness_plus) for f in fitness_plus]
    for _ in range(tam_pop):
        r = random.random()
        acumulado = 0
        j = -1
        while acumulado < r:
            j = (j + 1) % len(fitness)
            acumulado += fitness_rel[j]
        selecao.append(j)
    return selecao

def cross_mapeamento_parcial(c1: List[int], c2: List[int], p_cross: float) -> None:
    if p_cross <= 0:
        return

    r = random.random()
    if r > p_cross:
        return
    
    crossover_point1 = random.randint(0, len(c1)-1)
    crossover_point2 = random.randint(crossover_point1+1, len(c1))

    offspring1 = [-1] * len(c1)
    offspring2 = [-1] * len(c1)

    for i in range(crossover_point1, crossover_point2):
        offspring1[i] = c1[i]
        offspring2[i] = c2[i]

    def fill_remaining_offspring(offspring, parent, other_parent):
        for i in range(len(offspring)):
            if offspring[i] == -1:
                gene = parent[i]
                while gene in offspring:
                    index_in_other = other_parent.index(gene)
                    gene = parent[index_in_other]
                offspring[i] = gene

    fill_remaining_offspring(offspring1, c2, c1)
    fill_remaining_offspring(offspring2, c1, c2)

    for i in range(len(c1)):
        c1[i], c2[i] = offspring1[i], offspring2[i]

def mutar(c: List[int], p_mut: float) -> None:
    if p_mut <= 0:
        return

    for i1 in range(len(c)):
        if random.random() < p_mut:
            i2 = random.randint(0, len(c)-1)
            if i1 != i2:
                c[i1], c[i2] = c[i2], c[i1]

