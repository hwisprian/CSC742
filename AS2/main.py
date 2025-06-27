from chromosome import *
import matplotlib.pyplot as plt
from numpy.ma.extras import average
import pytest

##############################################################################################################
#    END of CHROMOSOME / Beginning of Genetic Algorithm Operations.
##############################################################################################################

# Q1.2 a function to generate a random initial population
def initialize_population(population_size, number_bits, number_variables):
    initial_population = []
    for _ in range(population_size):
        initial_population.append(Chromosome(None, string_length=number_bits, number_of_variables=number_variables))
    return initial_population

class ConvergenceException(Exception):
    """Exception raised for total fitness is zero values."""
    def __init__(self, message="Convergence"):
        self.message = message
        super().__init__(self.message)

#Q3 - Genetic Algorithm Operations - selection, crossover, mutation
#Q3.1 fitness proportionate selection
def selection(generation, fitness):
    min_fitness = min(fitness)
    # we have to worry about min fitness being negative to calculate probability :grimace:
    adjusted_fitness = []
    if min_fitness <= 0:
        # adjust to positive values for more accurate probabilities
        for fit in fitness:
            adjusted_fitness.append(fit + abs(min_fitness))
    else:
        for fit in fitness:
            adjusted_fitness.append(fit)
    total_fitness = sum(adjusted_fitness)

    if total_fitness == 0:
        raise ConvergenceException()

    # probability for each individual inverse so (1/fitness) / total fitness
    probabilities = [ 0 if fit == 0 else ((1 / fit) / total_fitness) for fit in adjusted_fitness]
    print("fitness", adjusted_fitness, "total fitness", total_fitness, "probabilities", probabilities)
    # randomly create a mating pool based on fitness probabilities
    return random.choices(generation, weights=probabilities, k=len(generation))
#Q3.2 and 3.3 see chromosome.py

#Q4.1 Genetic Algorithm Execution
def run_genetic_algorithm(population_size, number_generations, dejong_function, number_of_bits, number_of_variables):
    # init a bunch of things.
    parent_generation = initialize_population(population_size, number_of_bits, number_of_variables)
    # plot points is an array lists of points to plot
    #   first list is the best fitness for the generation
    #   second list is the average fitness for the generation
    plot_points = [[],[]]

    parent_fitness = get_fitness(parent_generation, dejong_function)
    # best this generation
    prev_best_fitness = min(parent_fitness)
    prev_best_chromosome = parent_generation[parent_fitness.index(prev_best_fitness)]

    # loop through reproduction (selection, crossover, mutation)
    try:
        for i in range(number_generations):
            children = reproduce(i, parent_generation, parent_fitness)
            child_fitness = get_fitness(children, dejong_function)
            best_fitness = min(child_fitness)
            # graph the best and average fitness per generation for each function
            best_chromosome = children[child_fitness.index(best_fitness)]
            plot_points[0].append(best_fitness)
            plot_points[1].append(average(child_fitness))
            # keep track if we have a new best!
            if best_fitness < prev_best_fitness:
                prev_best_chromosome = best_chromosome
                prev_best_fitness = best_fitness
            # the children become the parents
            parent_generation = children
            parent_fitness = child_fitness
    except ConvergenceException:
        print("Convergence...")
    plot_fitness(population_size, number_generations, dejong_function, prev_best_chromosome, number_of_variables, plot_points)

def get_fitness(generation, function):
    if function == 'f1':
        return [chrom.sphere_model_fitness() for chrom in generation]
    if function == 'f2':
        return [chrom.weighted_sphere_model_fitness() for chrom in generation]
    if function == 'f3':
        return [chrom.step_function_fitness() for chrom in generation]
    if function == 'f4':
        return [chrom.noisy_quartic_fitness() for chrom in generation]
    raise Exception("something happened! provide a fitness function!")

def reproduce(gen_num, parent_generation, fitness):
    # SELECTION!
    mating_pool = selection(parent_generation, fitness)
    children = []
    while len(children) < len(mating_pool):
        random_parents = random.sample(mating_pool, 2)
        # CROSSOVER!
        children.extend(random_parents[0].crossover(random_parents[1], gen_num, 2, .9))
        # MUTATION!
        children[0].mutation()
        children[1].mutation()
    return children

#Q4.2 Plot best fitness and average fitness per generation for each function
def plot_fitness(population_size, number_generations, dejong_function, best_chromosome, number_of_variables, plot_points):
    xdata = list(range(len(plot_points[0])))
    # Create the plot
    plt.plot(xdata, plot_points[0], color='red', label='best fitness', marker='+')
    plt.plot(xdata, plot_points[1], color='green',  label='avg fitness', marker='.')
    plt.scatter(best_chromosome.generation, best_chromosome.fitness, color='blue', label='best solution', marker='x')

    plt.legend(loc='best', fancybox=True, shadow=True, fontsize='small')

    # Add labels and title
    plt.xlabel("Generation")
    plt.ylabel("Fitness (y)")
    plt.title(f"Plot of Pop size {population_size} for {number_generations} Generations using {dejong_function} with {number_of_variables} vars")
    plt.figtext( .5, 0, f"Best Solution: {best_chromosome}", ha="center", va="bottom", fontsize=10)
    plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin to make space

    # Display the plot
    plt.show()

#Q4.3 Report the best solution and its decoded real values.

#################################################################
#                   Q4.1 Run with population size and generations
###################################################################

run_genetic_algorithm(20, 50, 'f1', 8, 2)

run_genetic_algorithm(20, 50, 'f2', 16, 3)

run_genetic_algorithm(30, 50, 'f3', 8, 4)

run_genetic_algorithm(50, 50, 'f4', 8, 4)

##################################################################################################
# --- Unit Tests are good for you ------------------------
#################################################################################################

def test_initialize_population():
    init_pop = initialize_population(10, 8, 4)
    assert len(init_pop) == 10
    assert len(init_pop[0].binary_strings) == 4
    assert len(init_pop[0].binary_strings[0]) == 8

def test_selection():
    mating_pool = selection([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
    assert len(mating_pool) == 6
    print("Mating Pool:", mating_pool)

    mating_pool = selection([1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6])
    assert len(mating_pool) == 6
    print("Mating Pool:", mating_pool)

    with pytest.raises(ConvergenceException, match="Convergence"):
        selection([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])