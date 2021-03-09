import numpy as np

ITERATION = 100


def de_rf(func, bounds, mut=0.8, crossp=0.9, popsize=60, its=ITERATION):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)

    # pdb.set_trace()
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff

    # convert from float to integer
    pop_denorm_convert = pop_denorm.tolist()

    result_list = []
    temp_list = []

    for index in pop_denorm_convert:
        temp_list.append(np.int(np.round_(index[0])))
        temp_list.append(np.int(np.round_(index[1])))
        temp_list.append(np.int(np.round_(index[2])))
        temp_list.append(np.int(np.round_(index[3])))
        temp_list.append(index[4])
        temp_list.append(np.int(np.round_(index[5])))
        result_list.append(temp_list)
        temp_list = []

    fitness = np.asarray([func(index[0], index[1], index[2], index[3], index[4], index[5])
                          for index in result_list])

    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + mut * (b - c)
            for i, v in enumerate(mutant):
                if 0 < v < 1: continue
                if v < 0: mutant[i] = v + 1
                if v > 1: mutant[i] = v - 1
            # mutant = np.clip(res, 0, 1)

            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trail_denorm_convert = trial_denorm.tolist()
            f = func(np.int(np.round_(trail_denorm_convert[0])), np.int(np.round_(trail_denorm_convert[1])),
                     np.int(np.round_(trail_denorm_convert[2])),
                     np.int(np.round_(trail_denorm_convert[3])), trail_denorm_convert[4],
                     np.int(np.round_(trail_denorm_convert[5])))

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def de_lr(func, bounds, mut=0.8, crossp=0.9, popsize=30, its=ITERATION):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)

    # pdb.set_trace()
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff

    # convert from float to integer
    pop_denorm_convert = pop_denorm.tolist()

    result_list = []
    temp_list = []

    for index in pop_denorm_convert:
        temp_list.append(index[0])
        temp_list.append(np.int(np.round_(index[1])))
        temp_list.append(index[2])
        result_list.append(temp_list)
        temp_list = []

    fitness = np.asarray([func(index[0], index[1], index[2])
                          for index in result_list])

    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + mut * (b - c)
            for i, v in enumerate(mutant):
                if 0 < v < 1: continue
                if v < 0: mutant[i] = v + 1
                if v > 1: mutant[i] = v - 1
            # mutant = np.clip(res, 0, 1)

            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trail_denorm_convert = trial_denorm.tolist()
            f = func(trail_denorm_convert[0], np.int(np.round_(trail_denorm_convert[1])),
                     trail_denorm_convert[2])

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def de_nb(func, bounds, mut=0.8, crossp=0.9, popsize=10, its=ITERATION):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)

    # pdb.set_trace()
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff

    # convert from float to integer
    pop_denorm_convert = pop_denorm.tolist()

    result_list = []
    temp_list = []

    for index in pop_denorm_convert:
        temp_list.append(index[0])
        result_list.append(temp_list)
        temp_list = []

#    print("*****************", result_list)
    fitness = np.asarray([func(index[0])
                          for index in result_list])

    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + mut * (b - c)
            for i, v in enumerate(mutant):
                if 0 < v < 1: continue
                if v < 0: mutant[i] = v + 1
                if v > 1: mutant[i] = v - 1
            # mutant = np.clip(res, 0, 1)

            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trail_denorm_convert = trial_denorm.tolist()
            f = func(trail_denorm_convert[0])

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def de_mlpn(func, bounds, mut=0.8, crossp=0.9, popsize=60, its=ITERATION):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)

    # pdb.set_trace()
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff

    # convert from float to integer
    pop_denorm_convert = pop_denorm.tolist()

    result_list = []
    temp_list = []

    for index in pop_denorm_convert:
        temp_list.append(index[0])
        temp_list.append(index[1])
        temp_list.append(index[2])
        temp_list.append(np.int(np.round_(index[3])))
        temp_list.append(index[4])
        temp_list.append(np.int(np.round_(index[5])))
        result_list.append(temp_list)
        temp_list = []

    fitness = np.asarray([func(index[0], index[1], index[2], index[3], index[4], index[5])
                          for index in result_list])

    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + mut * (b - c)
            for i, v in enumerate(mutant):
                if 0 < v < 1: continue
                if v < 0: mutant[i] = v + 1
                if v > 1: mutant[i] = v - 1
            # mutant = np.clip(res, 0, 1)

            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trail_denorm_convert = trial_denorm.tolist()
            f = func(trail_denorm_convert[0], trail_denorm_convert[1],
                     trail_denorm_convert[2],
                     np.int(np.round_(trail_denorm_convert[3])), trail_denorm_convert[4],
                     np.int(np.round_(trail_denorm_convert[5])))

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def de_knn(func, bounds, mut=0.8, crossp=0.9, popsize=20, its=ITERATION):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)

    # pdb.set_trace()
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff

    # convert from float to integer
    pop_denorm_convert = pop_denorm.tolist()

    result_list = []
    temp_list = []

    for index in pop_denorm_convert:
        temp_list.append(np.int(np.round_(index[0])))
        temp_list.append(np.int(np.round_(index[1])))
        result_list.append(temp_list)
        temp_list = []

    fitness = np.asarray([func(index[0], index[1])
                          for index in result_list])

    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + mut * (b - c)
            for i, v in enumerate(mutant):
                if 0 < v < 1: continue
                if v < 0: mutant[i] = v + 1
                if v > 1: mutant[i] = v - 1
            # mutant = np.clip(res, 0, 1)

            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trail_denorm_convert = trial_denorm.tolist()
            f = func(np.int(np.round_(trail_denorm_convert[0])), np.int(np.round_(trail_denorm_convert[1])))

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def de_smote(func, bounds, mut=0.8, crossp=0.9, popsize=30, its=ITERATION):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)

    # pdb.set_trace()
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff

    # convert from float to integer
    pop_denorm_convert = pop_denorm.tolist()

    result_list = []
    temp_list = []

    for index in pop_denorm_convert:
        temp_list.append(np.int(np.round_(index[0])))
        temp_list.append(index[1])
        temp_list.append(np.int(np.round_(index[2])))
        result_list.append(temp_list)
        temp_list = []

    fitness = np.asarray([func(index[0], index[1], index[2])
                          for index in result_list])

    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + mut * (b - c)
            for i, v in enumerate(mutant):
                if 0 < v < 1: continue
                if v < 0: mutant[i] = v + 1
                if v > 1: mutant[i] = v - 1
            # mutant = np.clip(res, 0, 1)

            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            trail_denorm_convert = trial_denorm.tolist()
            f = func(np.int(np.round_(trail_denorm_convert[0])), trail_denorm_convert[1],
                     np.int(np.round_(trail_denorm_convert[2])))

            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]
