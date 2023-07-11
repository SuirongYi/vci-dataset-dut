import random
import numpy as np
from deap import creator, base, tools
import matplotlib.pyplot as plt
import csv
import os
from scipy.stats import entropy
from Simulation import simulation_data, real_ped_data


pop_size = 100    # 族群规模
ngen = 200        # 迭代代数
cross_pro = 0.8   # 交叉概率
mutate_pro = 0.1  # 突变概率
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.dirname(path)
ped_csv_file = 'data\\trajectories_filtered\\intersection_01_traj_ped_filtered.csv'
veh_csv_file = 'data\\trajectories_filtered\\intersection_01_traj_veh_filtered.csv'
ped_file_path = os.path.join(path, ped_csv_file)
veh_file_path = os.path.join(path, veh_csv_file)

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))             # 最小化问题
creator.create('Individual', list, fitness=creator.FitnessMin)
para_low = [0.0, 0.0, 0.0, 0.0]
para_up = [5.0, 5.0, 5.0, 5.0]


def genInd(low, up):
    return [random.uniform(low[0], up[0]), random.uniform(low[1], up[1]),
            random.uniform(low[2], up[2]), random.uniform(low[3], up[3])]


toolbox = base.Toolbox()
toolbox.register('genInd', genInd, para_low, para_up)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.genInd)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


# 评价函数
def evaluate(ind):
    real_distribution = real_ped_data(ped_file_path)
    sim_distribution = simulation_data(ped_file_path, veh_file_path, list(ind))
    f = entropy(real_distribution, sim_distribution)
    return f,


toolbox.register('eval', evaluate)

# 记录迭代数据
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('avg', np.mean)
stats.register('min', np.min)
stats.register('std', np.std)
logbook = tools.Logbook()
logbook.header = ['gen', 'navels'] + stats.fields

# 注册遗传算法操作 - 选择,交叉,突变
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=20, low=para_low, up=para_up)
toolbox.register('mutate', tools.mutPolynomialBounded, eta=20, low=para_low, up=para_up, indpb=0.2)

pop = toolbox.population(pop_size)  # 生成初始族群

# 评价初始族群
invalid_ind = [ind for ind in pop if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.eval, invalid_ind)
for ind, fitness in zip(invalid_ind, fitnesses):
    ind.fitness.values = fitness
record = stats.compile(pop)
logbook.record(gen=0, nevals=len(invalid_ind), **record)

# 遗传算法迭代
for gen in range(1, ngen + 1):
    # 育种选择
    offspring = toolbox.select(pop, pop_size)  # 子代规模与父代相同
    offspring = [toolbox.clone(_) for _ in offspring]

    # 变异操作
    # 突变
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cross_pro:
            toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            del ind2.fitness.values
    for ind in offspring:
        if random.random() < mutate_pro:
            toolbox.mutate(ind)
            del ind.fitness.values

    # 评价子代
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.eval, invalid_ind)
    for ind, fitness in zip(invalid_ind, fitnesses):
        ind.fitness.values = fitness

    # 环境选择
    combinedPop = pop + offspring  # 采用精英策略,加速收敛
    pop = tools.selBest(combinedPop, pop_size)

    # 记录数据
    record = stats.compile(pop)
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
print(logbook)

# 输出结果
bestInd = tools.selBest(pop, 1)[0]
bestFit = bestInd.fitness.values[0]
print('最优解为: ' + str(bestInd))
print('函数最小值为: ' + str(bestFit))
minFit = logbook.select('min')
avgFit = logbook.select('avg')
plt.plot(minFit, 'b-', label='Minimum Fitness')
plt.plot(avgFit, 'r-', label='Average Fitness')
plt.xlabel('# Gen')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.show()
