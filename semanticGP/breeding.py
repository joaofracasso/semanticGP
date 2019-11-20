import random
from functools import partial
import datetime


from deap import tools


def varAnd(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        #off1, off2 = map(toolbox.clone, toolbox.selectMate(offspring))
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                              offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            #off1, off2 = toolbox.mate(off1, off2)
            #del off1.fitness.values, off2.fitness.values
    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
    return offspring

def eaSimple(population, toolbox, cxpb, mutpb, ngen, X_train, y_train,
                   X_test, y_test, stats=None, halloffame=None, verbose=__debug__):
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'exec_time', 'error_train', 'error_test', 'size'] + (stats.fields if stats else [])
    time_init = datetime.datetime.now()
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(partial(toolbox.evaluate , X=X_train, y=y_train), invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)
    time_end = datetime.datetime.now()
    exec_time = time_end - time_init
    record = stats.compile(population) if stats else {}
    error_train = toolbox.evaluate(halloffame[0], X=X_train, y=y_train)[0]
    error_test = toolbox.evaluate(halloffame[0], X=X_test, y=y_test)[0]
    logbook.record(gen=0, nevals=len(invalid_ind), exec_time=exec_time.total_seconds(),\
                   error_train=error_train, error_test=error_test, size=len(halloffame[0]) , **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        time_init = datetime.datetime.now()
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(partial(toolbox.evaluate , X=X_train, y=y_train), invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
        
        # Replace the current population by the offspring
        population[:] = offspring
        time_end = datetime.datetime.now()
        exec_time = time_end - time_init
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        error_train = toolbox.evaluate(halloffame[0], X=X_train, y=y_train)[0]
        error_test = toolbox.evaluate(halloffame[0], X=X_test, y=y_test)[0]
        logbook.record(gen=gen, nevals=len(invalid_ind), exec_time=exec_time.total_seconds(),\
                   error_train=error_train, error_test=error_test, size=len(halloffame[0]) , **record)
        if verbose:
            print(logbook.stream)

    return population, logbook