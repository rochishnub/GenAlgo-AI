
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

"""
Class to solve 8-Queens Puzzle
Inputs: Population size
    You can edit population size by changing value in solve_queens function
Output: State with best fitness

"""

class Improved_Queens():
    def __init__(self, popl_size):

        # initializing required parameters
        self.popl_size = popl_size              # population size
        self.population = self.generate_popl()  # list of population = 'xxxxxxxx'
        self.fitness_popl = self.generate_fit() # list of fitness of population
        self.delta_fit = 0                      # keeps count of best fitness over generations
        self.best_fit = max(self.fitness_popl)  # best fitness
        self.fit_graph = [self.best_fit]        # list of best fitness in each generation
        self.num_gen = 0                        # number of generations

    def fitness(self, state):
        cost = 0
        for i in range(8):
            for j in range(8):

                # for queens in same row
                if state[i]==state[j] and i!=j: 
                    cost += 1
                
                # for queens in same diagonal
                if (i-j)==(int(state[i])-int(state[j])) and i!=j:
                    cost += 1
                if (j-i)==(int(state[i])-int(state[j])) and i!=j:
                    cost += 1

        # subtracting cost/2 to take care of repetitions
        return 29-(cost/2)

    # generates random number x and fills population as 'xxxxxxxx'
    def generate_popl(self):
        row = random.randint(0,7)           
        state = str(row)*8
        population = [state]*self.popl_size
        return population
    
    # generates fitness from population
    def generate_fit(self):
        fitness_popl = [self.fitness(i) for i in self.population]
        return fitness_popl

    # one-point crossover
    def one_crossover(self, mom, dad):
        c = random.randint(1,7)
        son = mom[:c] + dad[c:]
        daughter = dad[:c] + mom[c:]
        return son, daughter

    # two-point crossover
    def two_crossover(self, mom, dad):
        c_1, c_2 = random.sample(range(8), 2)
        c1 = min(c_1, c_2)
        c2 = max(c_1, c_2)
        son = mom[:c1] + dad[c1:c2] + mom[c2:]
        daughter = dad[:c1] + mom[c1:c2] + dad[c2:]
        return son, daughter

    # main crossover function with 80% probability to one-point and rest to two-point
    def super_crossover(self, mom, dad):
        if random.randint(0,10)<8:
            return self.one_crossover(mom, dad)
        else:
            return self.two_crossover(mom, dad)

    # mutate function mutates one gene of a chromosome
    def mutate(self, child):
        c = random.randint(0,7)
        row = int(child[c])
        num = list(range(0,8))
        num.remove(row)
        rep = random.choice(num)
        child = child[:c] + str(rep) +child[c+1:]
        return child
    
    # main mutate function
    def super_mutate(self, child, order):
        order = order-5         # order is the number of row-clashes
        if order>1:             # if order is more than 5 mutation happens multiple times
            for _ in range(order):
                child = self.mutate(child)
        else:
            child = self.mutate(child)
        return child

    # main selection function
    def super_selection(self):
        parents = self.tour_selection()
        return parents

    # tournanment selection with k = 1/20 times size of population
    def tour_selection(self):
        tournament = random.sample(self.population, k=self.popl_size//20)
        tournament.sort(key=self.fitness, reverse=True)
        return tournament[0], tournament[1]
    
    # termination condition for genetic algorithm
    def terminate(self):
        # when best fitness reaches max =29
        if max(self.fitness_popl)==29:
            return True
        # best fitness gets stuck and repeats for 30 generations
        elif self.delta_fit==30:
            return True
        else: return False

    # calculates the number of row clashes in state
    def n_repetition(self, state):
        choose = '01234567'
        for i in state:
            choose = choose.replace(i,'')
        return len(choose)

    # main genetic algorithm
    def genetic_algo(self):
        while (not self.terminate()):
            # initialize new population as empty list
            new_popl = []       

            for _ in range(self.popl_size):
                mom, dad = self.super_selection()                   # selection
                son, daughter = self.super_crossover(mom, dad)      # crossover
                o_son = self.n_repetition(son)                      # row clashes in son
                o_daughter = self.n_repetition(daughter)            # row clashes in daughter
                
                # mutation
                # probability of mutation is proportional to row clashes
                if random.randint(0,7) < o_son+2:                   # mutation for son
                    son = self.super_mutate(son, o_son)             
                if random.randint(0,7) < o_daughter+2:              # mutation for daughter
                    daughter = self.super_mutate(daughter, o_daughter)
                
                # adding children to new population
                new_popl.append(son)
                new_popl.append(daughter)

            new_popl = list(set(new_popl))                          # makes distinct set
            new_popl.sort(key=self.fitness, reverse=True)           # sorts by fitness
            
            # update change in best fitness
            if max(new_popl,key=self.fitness)==max(self.population,key=self.fitness):
                self.delta_fit += 1
            else:
                self.delta_fit = 0
            
            # update required parameters for next generation
            self.population = new_popl[:self.popl_size]
            self.fitness_popl = self.generate_fit()
            self.best_fit = max(self.fitness_popl)
            self.fit_graph.append(self.best_fit)
            self.num_gen += 1

        # returns the state (chromosome) with best fitness
        return max(self.population, key=self.fitness)

"""
Class to solve Traveling Salesman Problem
Inputs: Population size
    You can edit population size by changing value in solve_tsp function
Output: Shortest path cycle without initial city

"""

class Improved_TSP():
    def __init__(self, popl_size):
        # initializing the map
        self.map = np.full((14,14), 10000) # initialize all distances as 10000
        self.map = pd.DataFrame(self.map)
        self.map.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
        self.map.index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']

        # no change to cells (x,x) because we will not need them for calculation
        # adding the distances in the map (multiplied by 100)
        self.add_city('a', 'g', 15)
        self.add_city('a', 'j', 20)
        self.add_city('a', 'l', 12)
        self.add_city('b', 'h', 19)
        self.add_city('b', 'i', 40)
        self.add_city('b', 'n', 13)
        self.add_city('c', 'd', 60)
        self.add_city('c', 'e', 22)
        self.add_city('c', 'f', 40)
        self.add_city('c', 'i', 20)
        self.add_city('d', 'f', 21)
        self.add_city('d', 'k', 30)
        self.add_city('e', 'i', 18)
        self.add_city('f', 'k', 37)
        self.add_city('f', 'l', 60)
        self.add_city('f', 'm', 26)
        self.add_city('f', 'n', 90)
        self.add_city('g', 'k', 55)
        self.add_city('g', 'l', 18)
        self.add_city('h', 'j', 56)
        self.add_city('h', 'n', 17)
        self.add_city('i', 'n', 60)
        self.add_city('j', 'l', 16)
        self.add_city('j', 'n', 50)
        self.add_city('k', 'm', 24)
        self.add_city('l', 'm', 40)

        # initializing required parameters
        self.popl_size = popl_size                  # population size
        self.population = self.generate_popl()      # population as 'abcdefghijklmn'

        self.fitness_popl = self.generate_fit()     # list of fitness in population
        self.best_fit = max(self.fitness_popl)      # best fitness of population
        self.fit_graph = [self.best_fit]            # list of best bitness in each generation

        self.cost_popl = self.generate_cost()       # list of costs in population
        self.best_cost = min(self.cost_popl)        # minimum cost in population
        self.cost_graph = [self.best_cost]          # list of minimum costs in each generation

        self.delta_cost = 0                         # keeps count of min cost over generations
        self.num_gen = 0                            # number of generations

    # adds city to map with distance
    def add_city(self, city1, city2, d):
        self.map[city1][city2] = d
        self.map[city2][city1] = d
    
    # calculates path cost
    def path_cost(self, path):
        path = path + path[0]       # append first city to complete cycle
        cost = 0
        for i in range(14):
            cost += self.distance(path[i], path[i+1])
        return cost
    
    # calculates fitness = 1/cost
    def fitness(self, path):
        return 1/self.path_cost(path)
    
    # finds distance between two cities from dataframe
    def distance(self, city1, city2):
        return self.map[city1][city2]
    
    # generates population as 'abcdefghijklmn'
    def generate_popl(self):
        path = 'abcdefghijklmn'
        population = [path]*self.popl_size
        return population
    
    # generates list of fitness from population
    def generate_fit(self):
        fitness_popl = [self.fitness(s) for s in self.population]
        return fitness_popl
    
    # generates list of cost from population
    def generate_cost(self):
        cost_popl = [self.path_cost(s) for s in self.population]
        return cost_popl

    # returns the combines adjacent matrix of parents
    def adj_matrix(self, mom, dad):
        d = {x: set() for x in mom}

        # finds neighbouring cities and add them to the set
        for i in range(14):
            d[mom[i]].add(mom[(i+1)%14])
            d[mom[i]].add(mom[i-1])
            d[dad[i]].add(dad[(i+1)%14])
            d[dad[i]].add(dad[i-1])
        
        # returns a dictionary with keys as cities and sets with their neighbours
        return d
    
    # deletes the neighbour from sets of all keys in adjacent matrix
    def delete_nbour(self, am, x):
        for key in am:
            am[key].discard(x)
        return am

    # edge recombination crossover
    def edge_recom_cross(self, mom, dad):
        parents = [mom, dad]
        children = []
        for p in parents:
            am = self.adj_matrix(mom, dad)      # creates adjacent matrix
            curr = p[0]
            child = ''
            for p in range(13):         # checks list of neighbours
                child += curr
                self.delete_nbour(am, curr)
                if len(am[curr])!=0:        # finds neighbour with smallest list
                    nxt = ''
                    min = 10
                    for x in am[curr]:
                        if len(am[x])<min:
                            nxt = x
                            min = len(am[x])
                        elif len(am[x])==min:
                            nxt += x
                    curr = random.choice(nxt)       # chooses random that becomes next neighbour
                else:
                    nxt = [key for key in am if key not in child]
                    curr = random.choice(nxt)
            children.append(child+curr)
        return children
    
    # ordered crossover function
    def ordered_cross(self, mom, dad):
        cut1, cut2 = random.sample(range(8), 2)
        c1 = min(cut1,cut2)
        c2 = max(cut1,cut2)         # selects two cut points
        mom_keep = mom[c1:c2]
        dad_keep = dad[c1:c2]
        mom_left = mom
        dad_left = dad              # genes remaining after cutting
        for i in range(c1, c2):
            mom_left = mom_left.replace(dad[i],'')
            dad_left = dad_left.replace(mom[i],'')
        
        # recombining to make children
        son = mom_left[:c1] + dad_keep + mom_left[c1:]          
        daughter = dad_left[:c1] + mom_keep + dad_left[c1:]
        return son, daughter

    # main crossover function that chooses between ERX and OX
    def super_crossover(self, mom, dad):
        if random.randint(0,1)==0:
            return self.ordered_cross(mom, dad)
        else: return self.edge_recom_cross(mom, dad)

    # main selection function
    def selection(self):
        #tournament style
        tournament = random.sample(self.population, k=self.popl_size//20)
        tournament.sort(key=self.fitness, reverse=True)
        return tournament[0], tournament[1]

    # 2-opt mutation function
    def two_opt_mutate(self, child):
        c_1, c_2 = random.sample(range(14), 2)
        c1 = min(c_1, c_2)
        c2 = max(c_1, c_2)
        temp = child[c1:c2]
        return child[:c1] + temp[::-1] + child[c2:]

    # 1-opt mutation function
    def one_opt_mutate(self, child):
        r = random.randint(0,13)
        temp = child[r:]
        return child[:r] + temp[::-1]
    
    # main mutation function that chooses between 1-opt and 2-opt
    def super_mutate(self, child):
        if random.randint(0,1)==0:
            return self.two_opt_mutate(child)
        else: return self.one_opt_mutate(child)
    
    # termination condition for the genetic algo
    def terminate(self):
        # if number of generations reach 100
        if self.num_gen==100:
            return True
        # if best cost does not change for 5 generations
        elif self.delta_cost==5:
            return True
        else: return False

    # main genetic algorithm
    def genetic_algo(self):
        while (not self.terminate()):
            # initializing new population as empty list
            new_popl = []

            for _ in range(self.popl_size):
                mom, dad = self.selection()                         # selection
                son, daughter = self.super_crossover(mom, dad)      # crossover
                if random.randint(0,100)<35:                        # mutation
                    son = self.super_mutate(son)
                    daughter = self.super_mutate(daughter)
                
                # add children to new population
                new_popl.append(son)
                new_popl.append(daughter)
            
            new_popl = list(set(new_popl)) + self.population[:10]   # create distinct set
            new_popl.sort(key=self.fitness, reverse=True)           # sort by fitness
            
            # update count of change in best cost
            if self.path_cost(new_popl[0])==self.best_cost:
                self.delta_cost += 1
            else:
                self.delta_cost = 0 

            # update required parameters
            self.population = new_popl[:self.popl_size]

            self.fitness_popl = self.generate_fit()
            self.best_fit = max(self.fitness_popl)
            self.fit_graph.append(self.best_fit)

            self.cost_popl = self.generate_cost()
            self.best_cost = min(self.cost_popl)
            self.cost_graph.append(self.best_cost)

            self.num_gen += 1
        
        # returns path with least cost or best fitness
        # does not include the first city
        return max(self.population, key=self.fitness)

# function to run 8-queens puzzle
def solve_queens():
    q = Improved_Queens(1000)                               # initialize problem
    print('Start State: ' + q.population[0])                # print start state 
    sol = q.genetic_algo()                                  # run genetic algorithm
    print('Solution: ' + sol)                               # print solution obtained
    print('Fitness: ' + str(q.best_fit))                    # print fitness of solution
    print('Number of generations: ' + str(q.num_gen))       # print number of generations needed
    plt.plot(q.fit_graph)                                   # plot graph of best fitness vs generation
    plt.xlabel('Number of Generations')
    plt.ylabel('Best Fitness')
    plt.xticks(range(0,q.num_gen+1))
    plt.show()                                      

# function to run traveling salesman problem
def solve_tsp():
    t = Improved_TSP(500)                                   # initialize problem
    print('Start State: ' + t.population[0])                # print start state
    sol = t.genetic_algo()                                  # run genetic algorithm
    print('Shortest Path: ' + sol + sol[0])                 # print shortest path obtained
    print('Shortest Path Cost: ' + str(t.best_cost/100))    # print cost of shortest path
    print('Number of generations: ' + str(t.num_gen-5))     # print generation number when it hits solution
    cost_g = [i/100 for i in t.cost_graph]
    plt.plot(cost_g)                                        # plot graph of min cost vs generation
    plt.xlabel('Number of Generations')
    plt.ylabel('Best Path Cost')
    plt.xticks(range(0,t.num_gen+1))
    plt.show()

# driver function
while True:
    s = input('Enter Q for Queens  |  T for Traveling Salesman \nWhich algo would you like to test: ')
    if s=='Q':
        solve_queens()
        break
    if s=='T':
        solve_tsp()
        break
    else:
        print('Please Enter Q or T')
