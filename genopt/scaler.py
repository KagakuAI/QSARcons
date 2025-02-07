def linear_scaling(pop):

    pop.calc_statistics()
    c = 1.2
    a = b = delta = 0.0

    pop_rawAvg = pop.stats["rawAvg"]
    pop_rawMax = pop.stats["rawMax"]
    pop_rawMin = pop.stats["rawMin"]
   
    if pop_rawAvg == pop.stats["rawMax"]:
        a = 1.0
        b = 0.0
    elif pop_rawMin > (c * pop_rawAvg - pop_rawMax / c - 1.0):
        delta = pop_rawMax - pop_rawAvg
        a = (c - 1.0) * pop_rawAvg / delta
        b = pop_rawAvg * (pop_rawMax - (c * pop_rawAvg)) / delta
    else:
        delta = pop_rawAvg - pop_rawMin
        a = pop_rawAvg / delta
        b = -pop_rawMin * pop_rawAvg / delta

    for i in range(len(pop)):
        f = pop[i].score
        f = f * a + b
        if f < 0:
            f = 0.0
        pop[i].fitness = f

    pop.calc_statistics()


def sigma_trunc_scaling(pop):

    pop.calc_statistics()

    c = 2
    
    rawAvg = pop.stats["rawAvg"]
    rawDev = pop.stats["rawDev"]
    
    for i in range(len(pop)):
        
        f = pop[i].score - rawAvg
        f += c * rawDev
        
        if f < 0: 
            f = 0.0

        pop[i].fitness = f

    pop.calc_statistics()