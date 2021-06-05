import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams["figure.figsize"] = (20,10)

class GA():
    def __init__(self, p_c=0.75, p_m=0.3, pop_size=30, len_of_indivs=33, iters=1000, maximization=True):
        self.maximization = maximization
        self.p_c = p_c
        self.p_m = p_m
        self.iters = iters

        self.pop_shape = (pop_size, len_of_indivs)
        self.pop = np.zeros(self.pop_shape)
        self.fitness = np.zeros(self.pop_shape[0])

        self.fitness_list = np.zeros((iters,))

    def run(self, repeat=30, show_result=False):
        '''
        run several time for stablize the result
        parameter:
        - repeat : times of runing the progress
        - show_result : print every iter result or not
        '''
        for i in range(1, repeat+1):
            print(f'\rnumber : {i}, best_fitness = {self.fitness_list[-1]}', end='')
            fitness = np.array(self.process(show_result))
            self.fitness_list += (fitness - self.fitness_list) / i
        print(f'\rnumber : {repeat}, best_fitness = {self.fitness_list[-1]}', end='')

    def process(self, show_result=False):
        # store the best score
        fitness_list = []

        self.initialization()
        fitness = self.evaluation(self.pop)

        idx = np.argsort(fitness)[-1]
        fitness_g_best = fitness[idx]
        indiv_g_best = self.pop[idx, :]

        for _iter in range(self.iters):
            next_gen = []

            for _ in range(int(self.pop_shape[0]/2)):
                i, j = self.rws(2, fitness)
                indiv_0, indiv_1 = self.pop[i, :].copy(), self.pop[j, :].copy()
                if np.random.rand() < self.p_c:
                    indiv_0, indiv_1 = self.crossover(indiv_0, indiv_1)

                    if np.random.rand() < self.p_m:
                        indiv_0 = self.mutation(indiv_0)
                        indiv_1 = self.mutation(indiv_1)

                next_gen.append(indiv_0)
                next_gen.append(indiv_1)

            pop = np.array(next_gen)
            fitness = self.evaluation(pop)

            if self.maximization:
                idx = np.argsort(fitness)[-1]
                if fitness[idx] > fitness_g_best: 
                    fitness_g_best = fitness[idx]
                    # indiv_g_best = pop[idx, :]
            else:
                idx = np.argsort(fitness)[0]
                fitness_g_best = fitness[idx]
                # indiv_g_best = pop[idx, :]


            if _iter % 1 == 0:
                fitness_list.append(fitness_g_best) 
                if show_result == True:
                    print('Gen {}:'.format(_iter))
                    print('The global best fitness:', fitness_g_best)
                    print('The global best individual:', indiv_g_best)
                    # print(fitness_g_best)
        return fitness_list


    def initialization(self): 
        self.pop = np.random.randint(low = 0, high = 2, size = self.pop_shape)

    def _fitness(self, arr):
        _str = ""
        for _ in range(0,18): 
            _str += str(arr[_])
        x1 = -3.0 + (int(_str, 2)) *((12.1-(-3.0))/(2**18-1))
        _str = ""
        for _ in range(18,33): 
            _str += str(arr[_])
        x2 = 4.1 + (int(_str, 2)) *((5.8-(4.1))/(2**15-1))
        return 21.5 + x1 * np.sin(4 * np.pi * x1) + x2 * np.sin(20 * np.pi * x2)

    def evaluation(self, pop):
        return np.array([self._fitness(i) for i in pop])

    def crossover(self, parent_0, parent_1):
        assert(len(parent_0) == len(parent_1))
        point = np.random.randint(len(parent_0))
        offspring_0 = np.hstack((parent_0[:point], parent_1[point:]))
        offspring_1 = np.hstack((parent_1[:point], parent_0[point:]))
        assert(len(offspring_0) == len(parent_0))
        return offspring_0, offspring_1

    def mutation(self, indiv):
        point = np.random.randint(len(indiv))
        indiv[point] = 1 - indiv[point]
        return indiv

    def rws(self, size, fitness):
        if self.maximization:
            fitness_ = fitness
        else:
            fitness_ = 1.0 / fitness
        idx = np.random.choice(np.arange(len(fitness_)), size=size, replace=True, p=fitness_/fitness_.sum())
        return idx

    def plot(self, label=None):
      plt.plot(self.fitness_list, label=label)
      plt.xlim(1, 1000)
      plt.xticks(np.arange(1, 1001, 100))
      plt.yticks(np.arange(30, 41, 5))


if __name__ == '__main__':
    # p_m differ experiment
    # for p_m in np.arange(0, 1.1, 0.1):
    #   print(f"\nSete p_m = {p_m}")
    #   ga = GA(p_m=p_m)
    #   ga.run(repeat=30, show_result=False)
    #   ga.plot(label=f'p_m={p_m}')
    #   if p_m == 0.5:
    #     plt.legend()
    #     plt.show()

    # plt.legend()
    # plt.show()
    
    # p_c differ experiment
    p_cs = np.round(np.arange(0.1, 1.01, 0.05) * 100) / 100
    for p_c in p_cs:
      print(f"\nSete p_c = {p_c}")
      ga = GA(p_c=p_c)
      ga.run(repeat=50, show_result=False)
      ga.plot(label=f'p_c={p_c}')
      if p_c == 0.5:
        plt.legend()
        plt.show()

    plt.legend()
    plt.show()