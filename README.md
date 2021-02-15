# GenAlgo-AI
Using Genetic Algorithms to solve the 8-Queens Puzzle and the Traveling Salesman Problem. The algorithm is written in Python 3.8.6.
There are two sets of algorithms for each problem, one simple and one optimised.

### Genetic Algorithms
Genetic Algorithms are search-based optimization techniques that draw inspiration from the biological principle of Natural selection.
A genetic algorithm consists of:
1. **Chromosome:** A representation of a state of the problem.
2. **Gene:** A particular element of part of the chromosome.
2. **Population:** A set of chromosomes that are input to the algorithm. They form the base.
4. **Fitness:** A function to evaluate a chromosome.

The genetic algorithm consists of the following steps:
1. Form the initial population.
2. **Selection:** Method to select the parents that will create children for the next generation.
3. **Crossover:** Method that reproduces the children
4. **Mutation:** Method to change or alter the child
5. **Survival:** Choose which children form the population for the next generation.
6. Repeat steps 2-6 until termination and return chromosome with best fitness.

### 8-Queens Puzzle
Aim: Find a configuration to place 8 queens on a chessboard such that no queens attack each other.
State: Assume initially all queens are in distinct columns. A state is the string representing the row containing a queen.
Fitness: The number of pairs of non-attacking queens + 1

### Traveling Salesman Problem
Aim: Find the shortest hamiltonian cycle in a graph
State: A string sequence of cities visited excluding first city.
Fitness: Path cost
!E:\3-2\AI\Gen_algo\Material\TSP_map.png
