import numpy as np
import math
import copy
import random

from deap import gp
from sklearn import linear_model

def replace(individual, node, subtree):
    slice_ = individual.searchSubtree(node)
    individual[slice_] = subtree
    return individual

def getnodesHeight(individual):
    """
    Get nodes height: get depth of all nodes in a s-expression

    :param individual: s-expressiom
    :returns: list of depth 
    """
    stack = [0]
    depthNodes = []
    for elem in individual:
        depth = stack.pop()
        depthNodes.append(depth)
        stack.extend([depth+1] * elem.arity)
    return depthNodes

def selectRandomNode(p):
    """Select Random Node: draws with uniform probability a random number r from the interval [1, height(p)],
    picks at random a node im p at depth r and returns it

    :param p: s-expression
    :returns: selected node
    """
    depthNodes = getnodesHeight(p)
    r = random.randint(1,max(depthNodes))
    nodes = []
    for idx, depth in enumerate(depthNodes):
        if depth == r:
            nodes.append(idx)
    return random.choice(nodes)

def buildSemanticLibrary(size_, toolbox, inSamples, metric):
    itens, semantics  = [], []
    firstItem = toolbox.item()
    itens.append(firstItem)
    semantics.append(toolbox.predict(firstItem , X=inSamples))
    while len(itens)< size_:
        newItem = toolbox.item()
        newSemantic = toolbox.predict(newItem , X=inSamples)
        flag = False
        for semantic in semantics:
            if metric(semantic, newSemantic) == 0:
                flag = True
                break
        if not(flag) :
            itens.append(newItem)
            semantics.append(newSemantic)
    return tuple([itens, semantics])


def buildLibrary(size_, toolbox, inSamples):
    itens = []
    itens.append(toolbox.item())
    while len(itens)< size_:
        newItem = toolbox.item()
        flag = False
        for item in itens:
            if item == newItem:
                flag = True
                break
        if not(flag) :
            itens.append(newItem)
    predicts = []
    for item in itens:
        predicts.append(toolbox.predict(item, inSamples=inSamples))
    return tuple([itens, predicts])

def buildLibraryIndividual(individual, creatorItem, size_, toolbox, inSamples):
    nodesInd = np.random.choice(len(individual), min(size_,len(individual)), replace=False)
    itens = []
    predicts = []
    for node in nodesInd:
        item = creatorItem(individual[individual.searchSubtree(node)])
        predict = toolbox.predict(item, inSamples)
        itens.append(item), predicts.append(predict)
    return tuple([itens, predicts])

def librarySearch(library, pset, sm, metric, *forbidden):
    """Library Search: a library of known programs is searched for a programs p' that minimizes the semantic distance to sm.

    Pawlak, T. P., & Krawiec, K. (2017). Competent Geometric Semantic Genetic Programming for Symbolic Regression and Boolean Function Synthesis.
    Evolutionary Computation, (x), 1–36. https://doi.org/doi:10.1162/EVCO_a_00205

    :param library: First expression participating
    :param sm: desired semantic
    :metric: metric to evaluete distance
    :returns: the closest item to desired from library
    """    
    minDistLib = float("inf")
    minDistERC = float("inf")
    nInfac = np.logical_not(np.logical_or(np.isnan(sm), np.isinf(sm))) 
    itens, values = library
    if sum(nInfac) == 0:
        return random.choice(itens)
    for idx, pred in enumerate(values):
        isForbidden = False
        itemValue = np.array(pred)
        dist = metric(itemValue[nInfac],sm[nInfac])
        if dist <= minDistLib:
            for f in forbidden:
                fValue = np.array(f)
                if np.array_equal(itemValue[nInfac],fValue[nInfac]):
                    isForbidden = True
                    break
            if not(isForbidden):
                pLib = itens[idx]
                minDistLib  = dist
    for value in np.linspace(-10**3, 10**3, 200):
        ercValue = value*np.ones(sm.shape)
        dist = metric(ercValue[nInfac],sm[nInfac])
        if dist <= minDistERC:
            for f in forbidden:
                fValue = np.array(f)
                if np.array_equal(ercValue[nInfac],fValue[nInfac]):
                    isForbidden = True
                    break
            if not(isForbidden):
                pErc = [gp.Terminal(value,True,pset.ret)]
                minDistERC = dist
    if minDistERC < minDistLib:
        p = pErc
    else:
        p = pLib
    return p

def computeLinearCombination(population, inSamples, desiredSamples, toolbox, alpha):
    predicts = [toolbox.predict(ind, inSamples) for ind in population]
    H = np.array(predicts)
    H = H.T
    reg = linear_model.Ridge(alpha = alpha)
    reg.fit(H, desiredSamples)
    #w = np.linalg.lstsq(H, desiredSamples)[0]
    return reg.coef_

def computeOutput(population, w, inSamples, toolbox):
    predicts = [toolbox.predict(ind, inSamples) for ind in population]
    H = np.array(list(predicts))
    H = H.T
    return np.matmul(H,w)

def searchExpr(expr, begin):
    """Return a slice object that corresponds to the
    range of values that defines the subtree which has the
    element with index *begin* as its root.
    """
    end = begin + 1
    total = expr[begin].arity
    while total > 0:
        total += expr[end].arity - 1
        end += 1
    return slice(begin, end)

def pos(expr, target):
    if expr[0].arity == 1:
        return 1
    else:
        slice_ = searchExpr(expr, 1)
        if slice_.start<=target and slice_.stop>target:
            return 1
        else:
            return 2

def child(expr,r):
    if r == 0:
        return expr,r
    slice_ = searchExpr(expr, 1)
    while slice_.stop <= r:
        slice_ = searchExpr(expr, slice_.stop)
    rangelist = range(slice_.start,slice_.stop)
    r = rangelist.index(r)
    expr = expr[slice_]
    return expr,r


def invert(a, k, o, c):
    aux = np.zeros(o.shape)
    if a == "add":
        aux = np.subtract(o,c)
        return aux
    elif a == "subtract":
        if k == 1:
            return np.add(o,c)
        elif k == 2:
            return np.subtract(c,o)
    elif a == "multiply":
        aux[~(abs(c) < 1e-15)] = np.divide(o[~(abs(c) < 1e-15)], c[~(abs(c) < 1e-15)])
        aux[~(abs(c) >= 1e-05)] = np.nan
        return aux
    elif a == "protectedDiv":
        if k == 1:
            aux[~(np.isinf(c))] = np.multiply(c[~(np.isinf(c))], o[~(np.isinf(c))])
            aux[np.logical_and(np.isinf(c), np.isclose(o, 0))] = np.nan
            aux[np.logical_and(np.isinf(c), ~(np.isclose(o, 0)))] = np.nan # it is not the correctly representation
            return aux
        elif k == 2:
            aux[~(abs(o) < 1e-15)] = np.divide(c[~(abs(o) < 1e-15)], o[~(abs(o) < 1e-15)])
            aux[~(abs(o) >= 1e-15)] = np.nan
            return aux
    elif a == "sin":
        aux[abs(o) <= 1] = np.arcsin(o[abs(o) <= 1])
        aux[abs(o) > 1] = np.inf # it is not the correctly representation
        return aux
    elif a == "cos":
        aux[abs(o) <= 1] = np.arccos(o[abs(o) <= 1])
        aux[abs(o) > 1] = np.inf # it is not the correctly representation
        return 
    elif a == "exp":
        aux[o > 0] = np.log(o[o > 0])
        aux[o <= 0] = np.inf # it is not the correctly representation
        return aux
    elif a == "log":
        aux[o < 1e2] = np.exp(o[o < 1e2])
        aux[o >= 1e2] = np.inf
        return aux
    elif a == "logical_not":
        return np.logical_not(o)
    elif a == "logical_and":
        aux[c] = o[c]
        aux[np.logical_and(np.logical_not(c),np.logical_not(o))] = np.nan
        aux[np.logical_and(np.logical_not(c),o)] = np.inf # it is not the correctly representation
        return aux
    elif a == "logical_or":
        aux[np.logical_not(c)] = o[np.logical_not(c)]
        aux[np.logical_and(c,o)] = np.nan
        aux[np.logical_and(c,np.logical_not(o))] = np.inf # it is not the correctly representation
        return aux
    elif a == "logical_nand":
        aux[c] = np.logical_not(o[c])
        aux[np.logical_and(np.logical_not(c),o)] = np.nan
        aux[np.logical_and(np.logical_not(c),np.logical_not(o))] = np.inf # it is not the correctly representation
        return aux
    elif a == "logical_nor":
        aux[np.logical_not(c)] = np.logical_not(o[np.logical_not(c)])
        aux[np.logical_and(c,np.logical_not(o))] = np.nan
        aux[np.logical_and(c,o)] = np.inf # it is not the correctly representation
        return aux

def semanticBackpropagation(desired, individual, node, creatorIndividual, inSamples, toolbox):
    a = copy.deepcopy(individual) 
    rAux = node
    d = copy.deepcopy(desired)
    nfac = ~(np.logical_or(np.isnan(d), np.isinf(d))) 
    while rAux != 0 and sum(nfac)!=0:
        k = pos(a,rAux)
        if a[0].arity == 2:
            if k==1:
                slice_ = searchExpr(a,1)
                c = toolbox.predict(creatorIndividual(a[slice_.stop:]), inSamples)
            elif k==2:
                slice_ = searchExpr(a,1)
                c = toolbox.predict(creatorIndividual(a[slice_]), inSamples)
            nInfac = ~(np.logical_or(np.isnan(d), np.isinf(d))) 
            c = np.array(c)
            d[nInfac] = invert(a[0].name,k ,d[nInfac],c[nInfac])
        elif a[0].arity == 1:
            nInfac = ~(np.logical_or(np.isnan(d), np.isinf(d))) 
            d[nInfac] = invert(a[0].name,1,d[nInfac],0)
        a, rAux = child(a,rAux)
        nfac = ~(np.logical_or(np.isnan(d), np.isinf(d))) 
    return d


def commonRegion(ind1, ind2):
    """Common Region: is the subset of program nodes that overlaps when program trees are superimposed on each other.

    :param ind1: First tree participating.
    :param ind2: Second tree participating.
    :returns: A tuple of two slices that node are in the common region.
    """
    index1 = []
    index2 = []
    i=0
    j=0
    while (len(ind1) > i) and (len(ind2) > j):
        if ind1[i].arity == ind2[j].arity:
            index1.append(i)
            index2.append(j)
            i = i+1
            j = j+1
        else:
            index1.append(i)
            index2.append(j)
            subtree1 = ind1.searchSubtree(i)
            subtree2 = ind2.searchSubtree(j)
            i = subtree1.stop
            j = subtree2.stop
    node = random.choice(tuple(zip(index1, index2)))
    slice1_ = ind1.searchSubtree(node[0])
    slice2_ = ind2.searchSubtree(node[1])
    return slice1_, slice2_
