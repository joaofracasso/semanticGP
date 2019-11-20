import copy
import random
import re
import numpy as np
from operator import eq, lt
from deap import gp
from tools.util import *

from sklearn import linear_model
from deap.tools.emo import sortNondominated
# Define the name of type for any types.
__type__ = object

######################################
# GP Crossover                      #
######################################

def sx(ind1, ind2):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    node1 = selectRandomNode(ind1)
    node2 = selectRandomNode(ind2)
    slice1 = ind1.searchSubtree(node1)
    slice2 = ind2.searchSubtree(node2)
    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
    return ind1, ind2

######################################
# SGP Crossovers                      #
######################################

def sdx(ind1, ind2, metric, sSensitivity, crossoverOperator, inSamples, toolbox):
    """Semantically Driven Crossover: exchange semantically equivalents subtrees, 
    resulting in children that are identical to their parents.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :param points: set of point on the doamin
    :param sSensitivity: Semantic Equivalence
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    i = 0
    while i < 5:
        p1, p2 = copy.deepcopy(ind1), copy.deepcopy(ind2)
        semantic1, semantic2 = toolbox.predict(p1, inSamples), toolbox.predict(p2, inSamples)      
        crossoverOperator(p1, p2)
        semanticOff1, semanticOff2 = toolbox.predict(p1, inSamples), toolbox.predict(p2, inSamples)
        if not(metric(semantic1, semanticOff1) < sSensitivity and
               metric(semantic2, semanticOff2) < sSensitivity and
               metric(semantic1, semanticOff2) < sSensitivity and
               metric(semantic2, semanticOff1) < sSensitivity):
            return p1, p2
        i += 1
    return p1, p2

def sax(ind1, ind2, metric, sSensitivity, creatorIndividual, inSamples, toolbox):
    """Semantically Aware Crossover: exchange semantically equivalents subtrees, 
    resulting in children that are identical to their parents.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :param points: set of point on the doamin
    :param sSensitivity: Semantic Equivalence
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    node1 = selectRandomNode(ind1)
    node2 = selectRandomNode(ind2)
    slice1 = ind1.searchSubtree(node1)
    slice2 = ind2.searchSubtree(node2)
    semantic1 = toolbox.predict(creatorIndividual(ind1[slice1]), inSamples)
    semantic2 = toolbox.predict(creatorIndividual(ind2[slice2]), inSamples)
    if metric(semantic1, semantic2) < sSensitivity:
        node1 = selectRandomNode(ind1)
        node2 = selectRandomNode(ind2)
        slice1 = ind1.searchSubtree(node1)
        slice2 = ind2.searchSubtree(node2)
    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
    return ind1, ind2

def ssx(ind1, ind2, metric, creatorIndividual, maxTrial, minSimilarity, maxSimilarity, inSamples, toolbox):
    """Semantically Similarity based Crossover: exchange semantically equivalents subtrees, 
    resulting in children that are identical to their parents.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :param points: set of point on the doamin
    :param max_Trial: number of trials to find a semantically similar pair
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    count = 0
    while (count < maxTrial):
        node1 = selectRandomNode(ind1)
        node2 = selectRandomNode(ind2)
        slice1 = ind1.searchSubtree(node1)
        slice2 = ind2.searchSubtree(node2)
        semantic1 =  toolbox.predict(creatorIndividual(ind1[slice1]), inSamples)
        semantic2 =  toolbox.predict(creatorIndividual(ind2[slice2]), inSamples)
        distance = metric(semantic1, semantic2)
        if (distance > minSimilarity) and (distance < maxSimilarity):
            ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
            return ind1, ind2
        else:
            count = count + 1
    ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
    return ind1, ind2

def mssx(ind1, ind2, metric, creatorIndividual, maxTrial, inSamples, toolbox):
    """Semantically Similarity based Crossover: exchange semantically equivalents subtrees,
     resulting in children that are identical to their parents.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :param points: set of point on the doamin
    :param max_Trial: number of trials to find a semantically similar pair
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    alpha = 0.001
    count = 0
    slice1_, slice2_  = None, None
    max_ = float("inf")
    while (count < maxTrial):
        node1 = selectRandomNode(ind1)
        node2 = selectRandomNode(ind2)
        slice1 = ind1.searchSubtree(node1)
        slice2 = ind2.searchSubtree(node2)
        semantic1 =  toolbox.predict(creatorIndividual(ind1[slice1]), inSamples)
        semantic2 =  toolbox.predict(creatorIndividual(ind2[slice2]), inSamples)
        sd = metric(semantic1, semantic2)
        if (sd >= alpha and sd <= max_):
            slice1_, slice2_ = slice1, slice2
            max_ = sd
        count = count + 1
    if slice1_ is None and slice2_ is None:
        slice1_, slice2_ = slice1, slice2
    ind1[slice1_], ind2[slice2_] = ind2[slice2_], ind1[slice1_]
    return ind1, ind2

######################################
# GSGP Crossovers                    #
######################################

def klx(ind1, ind2, k, metric, crossoverOperator, inSamples, toolbox):
    """ Krawiec and Lichocki Geometric Crossover (KLX): relies o an arbitrary base crossover operator to generato an approximately geometric offspring

    [1] Krawiec, Krzysztof and Lichocki, Pawel. "Approximating Geometric Crossover in Semantic Space"

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :param metric: distance metric 
    :param crossoverOperator: base crossover operator
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    bestValue1, bestValue2  = float('inf'), float('inf')
    off1, off2 = [], []
    semantic1, semantic2 = toolbox.predict(ind1, inSamples), toolbox.predict(ind2, inSamples)   
    for _ in range(k/2):
        pool1, pool2 = crossoverOperator(copy.deepcopy(ind1), copy.deepcopy(ind2))
        predict1, predict2 = toolbox.predict(pool1), toolbox.predict(pool2)   
        value1 = metric(semantic1, predict1) + metric(predict1, semantic2) + abs(metric(semantic1, predict1) - metric(predict1, semantic2))
        value2 = metric(semantic1, predict2) + metric(predict2, semantic2) + abs(metric(semantic1, predict2) - metric(predict2, semantic2))
        if value1 < bestValue1:
            off2 = copy.deepcopy(off1)
            off1 = copy.deepcopy(pool1)
            bestValue1 = value1
        elif value1 < bestValue2:
            off2 = copy.deepcopy(pool1)
            bestValue2 = value1
        if value2 < bestValue1:
            off2 = copy.deepcopy(off1)
            off1 = copy.deepcopy(pool2)
            bestValue2 = bestValue1
            bestValue1 = value2
        elif value2 < bestValue2:
            off2 = copy.deepcopy(pool2)
            bestValue2 = value2
    return off1, off2

def lgx(ind1, ind2, library, metric, creatorIndividual, inSamples, toolbox):
    """locally geometric semantic crossover (LGX): has controlled impact on program semantics and aims at making the offspring programs
     semantically more similar to each other than the parent programs [1].
     
    [1] Krawiec, Krzysztof; Pawlak, Tomasz. "Locally geometric semantic crossover: A study on the roles of semantics and homology in recombination operators" (2013)
    
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :population: library of (sub)programs
    :returns: A tuple of two trees.
     """
    slice1_,slice2_ = commonRegion(ind1, ind2)
    semantic1 =  toolbox.predict(creatorIndividual(ind1[slice1]), inSamples)
    semantic2 =  toolbox.predict(creatorIndividual(ind2[slice2]), inSamples)
    semantic1, semantic2 = np.array(semantic1), np.array(semantic2)
    sm = (semantic1 + semantic2) / 2
    pLine = librarySearch(library, sm, metric)
    ind1[slice1_], ind2[slice2_] = pLine , pLine
    return ind1, ind2

def cx(ind1, ind2, library, pset, metric, creatorIndividual, inSamples, toolbox):
    """
    Approximately Geometric Semantic Crossover (AGX): replaces subtrees in parents with such code fragments (subtrees) that the semantics 
    of offspring lays in the middle of the segment connecting parents semantics 
    
    [1] Krawiec, Krzysztof; Pawlak, Tomasz. "Approximating Geometric Crossover by Semantic Backpropagation"

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :population: library of (sub)programs
    :returns: A tuple of two trees.

    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    semantic1, semantic2 = toolbox.predict(ind1, inSamples), toolbox.predict(ind2, inSamples) 
    semantic1, semantic2 = np.array(semantic1), np.array(semantic2)  
    sm = (semantic1 + semantic2) / 2
    ind1, = cm(ind1, sm, library, pset, metric, creatorIndividual, inSamples, toolbox)
    ind2, = cm(ind2, sm, library, pset, metric, creatorIndividual, inSamples, toolbox)
    return ind1, ind2

def sgx(ind1, ind2, pset, creatorIndividual):
    for op in pset.primitives[pset.ret]:
        if op.name == "add":
            pAdd = op
        elif op.name == "subtract":
            pSub = op
        elif op.name == "multiply":
            pMul = op
    term1 = gp.Terminal(1,True,pset.ret)
    rand_ = gp.Terminal(random.random(),True,pset.ret)
    expr1, expr2 = [], []
    expr1.extend((pAdd, pMul, rand_)), expr2.extend((pAdd, pMul, rand_))
    expr1.extend(ind1), expr2.extend(ind2)
    expr1.extend((pMul, pSub, term1, rand_)), expr2.extend((pMul, pSub, term1, rand_))
    expr1.extend(ind2), expr2.extend(ind1)
    return creatorIndividual(expr1), creatorIndividual(expr2)

def mosax(ind1, ind2, metric, maxSizeLib, creatorIndividual, creatorItem, inSamples, toolbox):
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    library1 = buildLibraryIndividual(ind1, creatorItem, maxSizeLib, toolbox, inSamples)
    library2 = buildLibraryIndividual(ind2, creatorItem, maxSizeLib, toolbox, inSamples)
    ind1, = mosam(ind1, library2, metric, creatorIndividual, inSamples, toolbox)
    ind2, = mosam(ind2, library1, metric, creatorIndividual, inSamples, toolbox)
    return ind1, ind2

def modx(ind1, ind2, library, metric, creatorIndividual, creatorItem, inSamples, toolbox):
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    semantic1, semantic2 = toolbox.predict(ind1, inSamples), toolbox.predict(ind2, inSamples)
    semantic1, semantic2 = np.array(semantic1), np.array(semantic2)
    alpha = random.random()
    sm = (semantic1 + semantic2) / 2
    ind1, = modo(ind1, sm, library, metric, creatorIndividual, inSamples, toolbox)
    ind2, = modo(ind2, sm, library, metric, creatorIndividual, inSamples, toolbox)
    return ind1, ind2

def grasx(ind1, ind2, alpha, metric, maxSizeLib, creatorIndividual, creatorItem, inSamples, toolbox):
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    library1 = buildLibraryIndividual(ind1, creatorItem, maxSizeLib, toolbox, inSamples)
    library2 = buildLibraryIndividual(ind2, creatorItem, maxSizeLib, toolbox, inSamples)
    ind1, = grasm(ind1, alpha, library2, metric, creatorIndividual, inSamples, toolbox)
    ind2, = grasm(ind2, alpha, library1, metric, creatorIndividual, inSamples, toolbox)
    return ind1, ind2


######################################
# GP Mutations                       #
######################################

def sm(individual, toolbox):
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    node = selectRandomNode(individual)
    slice_ = individual.searchSubtree(node)
    individual[slice_] = toolbox.expr_mut()
    return individual,

######################################
# SGP Mutations                      #
######################################

def sdm(individual, metric, sSensitivity, mutationOperator, inSamples, toolbox):
    """Semantic Driven Mutation
    A mutation point randomly chosen in the parent and a new subtree is stochastically.
    Then the semantic equivalence is checked to determine if these two subtree (replaced and replacing subtrees in the mutatuin operation) are equivalent.
    If they are not semantically equivalent, the mutation is performed by simply replacing the subtree at the mutatuion point with the new generate subtree.abs

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when called.
    :param points: set of point on the doaminmeasuringSemantics
    :param sSensitivity: Semantic Equivalence
    :returns: A tuple of one tree.measuringSemantics
    """
    i = 0
    while (i < 5):
        p = copy.deepcopy(individual)
        semantic = toolbox.predict(p, inSamples) 
        mutationOperator(p, toolbox)
        semanticOff = toolbox.predict(p, inSamples)
        if not(metric(semantic, semanticOff) < sSensitivity):
            return p,
        i += 1
    return p,

def sam(individual, metric, sSensitivity, creatorIndividual, inSamples, toolbox):
    """Semantic Aware Mutation
    A mutation point randomly chosen in the parent and a new subtree is stochastically.
    Then the semantic equivalence is checked to determine if these two subtree (replaced and replacing subtrees in the $
    If they are not semantically equivalent, the mutation is performed by simply replacing the subtree at the mutatuion$

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when called.
    :param points: set of point on the doaminmeasuringSemantics
    :param sSensitivity: Semantic Equivalence
    :returns: A tuple of one tree.measuringSemantics
    """
    node = selectRandomNode(individual)
    slice_ = individual.searchSubtree(node)
    indEval = creatorIndividual(individual[slice_])
    semantic1 = toolbox.predict(indEval, inSamples)
    st2expr = toolbox.expr_mut()
    st2 = creatorIndividual(st2expr)
    semantic2 = toolbox.predict(st2, inSamples)
    sd = metric(semantic1, semantic2)
    if (sd < sSensitivity):
        node = selectRandomNode(individual)
        slice_ = individual.searchSubtree(node)
        st2expr = toolbox.expr_mut()
    individual[slice_] = st2expr
    return individual,


def ssm(individual, metric, creatorIndividual, maxTrial, minSimilarity, maxSimilarity, inSamples, toolbox):
    """Semantically Similarity based mutation:
    A mutation point randomly chosen in the parent and a new subtree is stochastically.
    Then the semantic equivalence is checked to determine if these two subtree (replaced and replacing subtrees in the $    If they are not semantically equivalent, the mutation is performed by simply replacing the subtree at the mutatuion$
    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when called.
    :param max_Trial: number of trials to find a semantically similar pair
    :returns: A tuple of one tree.
    """
    count = 0
    while (count < maxTrial):
        node = selectRandomNode(individual)
        slice_ = individual.searchSubtree(node)
        indEval = creatorIndividual(individual[slice_])
        semantic1 = toolbox.predict(indEval, inSamples)
        st2expr = toolbox.expr_mut()
        st2 = creatorIndividual(st2expr)
        semantic2 = toolbox.predict(st2, inSamples)
        distance = metric(semantic1, semantic2)
        if (distance > minSimilarity) and (distance < maxSimilarity):
            individual[slice_] = st2expr
            return individual,
        else:
            count = count + 1
    individual[slice_] = st2expr
    return individual,

######################################
# GSGP Mutations                     #
######################################

def cm(individual, desiredSementic, library, pset, metric, creatorIndividual, inSamples, toolbox):
    """
    Competent Mutation (CM): Search Operator. Parameters: t: Target Semantics, p: Parent Tree, L: 
    Library of Programs. Replaces the Subtree Rooted in Node n in Tree p With the Tree p
    
    Pawlak, T. P., & Krawiec, K. (2017). Competent Geometric Semantic Genetic Programming for Symbolic Regression and Boolean Function Synthesis.
    Evolutionary Computation, (x), 1–36. https://doi.org/doi:10.1162/EVCO_a_00205

    :param individual: tree participating in the mutation.
    :population: library of (sub)programs
    :metric: operator to calculate distance between semantics
    :returns: A tuple of two trees.

    """
    node = selectRandomNode(individual)
    D = semanticBackpropagation(desiredSementic.flatten(), individual, node,  creatorIndividual, inSamples, toolbox)
    semantic = np.array(toolbox.predict(individual, inSamples))
    D0 = semanticBackpropagation(semantic.flatten(), individual, node,  creatorIndividual, inSamples, toolbox)
    pLine = librarySearch(library, pset, D, metric, D0)
    slice_ = individual.searchSubtree(node)
    individual[slice_]= pLine[:]
    return individual,

def sgm(individual, step, pset, creatorIndividual, toolbox):
    for op in pset.primitives[pset.ret]:
        if op.name == "add":
            pAdd = op
        elif op.name == "subtract":
            pSub = op
        elif op.name == "multiply":
            pMul = op
        st1expr = toolbox.expr_mut(pset=pset)
        st2expr = toolbox.expr_mut(pset=pset)
        stepExpr = gp.Terminal(step,True, pset.ret)
    expr = []
    expr.extend((pAdd, pMul, stepExpr, pSub))
    expr.extend(st1expr)
    expr.extend(st2expr)
    expr.extend(individual)
    return creatorIndividual(expr),

def grasm(individual, alpha, library, metric, creatorIndividual, inSamples, toolbox):
    """

    :param individual: tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :population: library of (sub)programs
    :returns: A tuple of two trees.

    """
    node = selectRandomNode(individual)
    slice_ = individual.searchSubtree(node)
    indEval = creatorIndividual(individual[slice_])
    semantic = toolbox.predict(indEval, inSamples)
    itens, values = library
    maxCost = -float("inf")
    minCost =  float("inf")
    RCL = []
    for pred in values:
        deltaCost = metric(pred ,semantic)
        if deltaCost <= minCost and deltaCost >= 1e-5:
            minCost = deltaCost
        if deltaCost > maxCost:
            maxCost = deltaCost
    for idx, pred in enumerate(library[1]):
        deltaCost = metric(pred ,semantic)
        if deltaCost <= (minCost + alpha*(maxCost - minCost)) and deltaCost >= minCost:
            RCL.append(idx)
    pLine = itens[random.choice(RCL)]
    individual[slice_]= pLine[:]
    return individual,

def mosam(individual, library, metric, creatorIndividual, inSamples, toolbox):
    """
    Multi-objective aware mutation
    :param individual: tree participating in the mutation.
    :population: library of (sub)programs
    :creatorIndividual
    :
    :
    :returns: A tuple of two trees.

    """
    node = selectRandomNode(individual)
    slice_ = individual.searchSubtree(node)
    indEval = creatorIndividual(individual[slice_])
    semantic = toolbox.predict(indEval, inSamples)
    itens, values = library
    nonZeroItens = []
    for item, value in zip(itens, values):
        dist = metric(value,semantic)
        size_ = len(item)
        item.fitness.values = (-dist, -size_)
        if dist >= 1e-5:
            nonZeroItens.append(item)
    nonDominateLib = sortNondominated(nonZeroItens, len(itens), first_front_only=True)
    pLine = random.choice(nonDominateLib[-1])
    individual[slice_]= pLine[:]
    return individual,

def modm(individual, desiredSementic, library, metric, creatorIndividual, inSamples, toolbox):
    """

    :param individual: tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :population: library of (sub)programs
    :returns: A tuple of two trees.

    """
    node = selectRandomNode(individual)
    D = semanticBackpropagation(desiredSementic.flatten(), individual, node,  creatorIndividual, inSamples, toolbox)
    nInfac = np.logical_not(np.logical_or(np.isnan(D), np.isinf(D))) 
    if sum(nInfac) == 0:
        slice_ = individual.searchSubtree(node)
        pLine =random.choice(library[0])
        individual[slice_]= pLine[:]
        return individual,
    itens, values = library
    for item, value in zip(itens, values):
        itemValue = np.array(value)
        dist = metric(itemValue[nInfac],D[nInfac])
        size_ = len(item)
        item.fitness.values = (-dist, -size_)
    nonDominateLib = sortNondominated(itens, len(itens), first_front_only=True)
    pLine = random.choice(nonDominateLib[-1])
    slice_ = individual.searchSubtree(node)
    individual[slice_]= pLine[:]
    return individual,
