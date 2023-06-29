import random
import numpy as np
from deap import base, creator, tools

LOW, UP = -3, 3
ETA = 20
LENGTH_CHROM = 3

POPULATION_SIZE = 500
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATION = 100

RAND_SEED = 0
random.seed(RAND_SEED)

def eval_func(individual) -> tuple:
    x, y, z = individual
    return 1/(1 + (x-2) ** 2 + (y+1) ** 2 + (z-1) ** 2),

def random_point(a, b):
    return [random.uniform(a, b), random.uniform(a, b), random.uniform(a, b)]

def create_toolbox():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("randomPoint", random_point, LOW, UP)
    toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    toolbox.register('evaluate', eval_func)

    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
    toolbox.register('mutate', tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0 / LENGTH_CHROM)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('min', np.min)
    stats.register('avg', np.mean)

    return toolbox

def main():
    toolbox = create_toolbox()
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    #num_generations = 100
    print("\nEvol process")
    fitnesses = list(map(toolbox.evaluate, population))
    for indiv, ftn in zip(population, fitnesses):
        indiv.fitness.values = ftn

    print('\n', len(population), 'indivs eval\'d')
    for g in range(MAX_GENERATION):
        print(f'\n---- {g}-th gen ----')
        offspring = toolbox.select(population, len(population))

        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.values]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print(len(invalid_ind), 'indivs eval\'d')
        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]

        length = len(population)
        avg = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - avg ** 2) ** 0.5
        print('min =', min(fits), ' max =', max(fits))
        print('avg = ', round(avg, 2), ' dev = ', round(std, 2))
    print('\n---- End of evol -----')
    best_ind = tools.selBest(population, 1)[0]
    print('\nBest indiv', best_ind)
    print('\nNum of 1s', sum(best_ind))

if __name__ == "__main__":
        main()