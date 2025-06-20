from sqlite3.dbapi2 import paramstyle
from weakref import finalize

import matplotlib.pyplot as plt
import numpy as numpy
import random
import pytest

class Chromosome:
    binary_string = ''
    #Q1 Constructor, takes a binary string or generate a random binary string if one is not provided.
    def __init__(self, string_in, binary_string_length):
        if string_in is None or len(string_in) != binary_string_length:
            # if no binary string is provided, or it is the wrong length, generate a random one.
            for _ in range(binary_string_length):
                self.binary_string += str(random.randint(0,1))
        else:
            self.binary_string = string_in

    def get_accuracy(self, lower_bound, upper_bound):
        return (upper_bound - lower_bound) / (  pow(2,len(self.binary_string)) - 1 )

    def get_decimal_value(self):
        return int(self.binary_string, 2)

    #Q2.1 decode a binary chromosome into a real number based on lower and upper bound
    def get_real_value(self, lower_bound, upper_bound):
        accuracy = self.get_accuracy(lower_bound, upper_bound)
        return lower_bound + (accuracy * self.get_decimal_value())

    # Q2.2 implement evaluation functions for all the De Jong functions
    def sphere_model_fitness(self, number):
        fitness = 0
        for _ in range(number):
            xi = pow(self.get_real_value(-5.12, 5.12), 2)
            fitness += xi
        return fitness

    # Q2.2 implement evaluation functions for all the De Jong functions
    def weighted_sphere_model_fitness(self):
        fitness = 0
        for _ in range(number):
            xi = pow(self.get_real_value(-5.12, 5.12), 2)
            fitness += xi
        return fitness

    # Q2.2 implement evaluation functions for all the De Jong functions
    def step_function_fitness(self):
        pass

    # Q2.2 implement evaluation functions for all the De Jong functions
    def noisy_quartic_fitness(self):
        pass

    # Q3.2 one or two point cross over
    def crossover(self, other_parent, number_of_points, probability):
        string_length = len(self.binary_string)
        offspring_binary_string1 = ''
        offspring_binary_string2 = ''
        do_crossover = random.random() < probability
        if do_crossover:
            # TODO implement more than one point.
            point = random.randint(0, string_length)
            for i in range(string_length):
                if i < point:
                    offspring_binary_string1 += self.binary_string[i]
                    offspring_binary_string2 += other_parent.binary_string[i]
                else:
                    offspring_binary_string1 += other_parent.binary_string[i]
                    offspring_binary_string2 += self.binary_string[i]
        else:
            offspring_binary_string1 = self.binary_string
            offspring_binary_string2 = other_parent.binary_string
        children = []
        if string_length == 8:
            children.append(Chromosome8Bit(offspring_binary_string1))
            children.append(Chromosome8Bit(offspring_binary_string2))
        return children

    #Q3.3 bitwise mutation
    def mutation(self):
        offspring = ''
        probability = 1/len(self.binary_string)
        for bit in self.binary_string:
            new_bit = bit
            if random.random() < probability:
                new_bit = '1' if bit == '0' else '1' ## flip a bit
            offspring += new_bit
        return offspring

# ----- Chromosome subclasses with various set lengths why not ----

# Q1 - Chromosome class encode variables as 8bit binary string
class Chromosome8Bit(Chromosome):
    binary_string_length = 8
    def __init__(self, string_in):
        # no binary string was provided so lets use the Chromosome constructor to generate a random one.
        super().__init__(string_in, self.binary_string_length)

# Q1 - Chromosome class encode variables as 8bit binary string
class Chromosome16Bit(Chromosome):
    binary_string_length = 16
    def __init__(self, string_in):
        super().__init__(string_in, self.binary_string_length)

#Q1.2 a function to generate a random initial population
def initialize_population(population_size, number_bits):
    initial_population = []
    if number_bits == 8:
        for _ in range(population_size):
            initial_population.append(Chromosome8Bit(None))
    if number_bits == 16:
        for _ in range(population_size):
            initial_population.append(Chromosome16Bit(None))
    return initial_population

#Q3 - Genetic Algorithm Operations
#Q3.1 fitness proportionate selection
def selection(generation, fitness):
    total_fitness = sum(fitness)
    # probability for each individual is fitness / total fitness
    probabilities = [fit/total_fitness for fit in fitness]
    # randomly create a mating pool based on fitness probabilities
    return random.choices(generation, weights=probabilities, k=len(generation))

#Q4.1 Genetic Algorithm Execution
def run_genetic_algorithm(population_size, number_generations):
    parent_generation = initialize_population(population_size, 8)
    fitness = [chrom.sphere_model_fitness(3) for chrom in parent_generation]
    plot_fitness(0, parent_generation, fitness)
    for i in range(number_generations):
        new_generation = reproduce(parent_generation)
        new_fitness = [chrom.sphere_model_fitness(3) for chrom in new_generation]
        plot_fitness(i+1, new_generation, new_fitness)

def reproduce(parent_generation):
    fitness = [chrom.sphere_model_fitness(3) for chrom in parent_generation]
    mating_pool = selection(parent_generation, fitness)
    children = []
    while len(children) < len(mating_pool):
        random_parents = random.sample(mating_pool, 2)
        children.extend(random_parents[0].crossover(random_parents[1], 1, .9))
    # TODO mutate some children!
    return children

#Q4.2 Plot best fitness and average fitness per generation for each function
def plot_fitness(generation_number, generation, fitness):
    x_data = [chrom.get_decimal_value() for chrom in generation]
    #x_data = [chrom.get_real_value(-5.12, 5.12) for chrom in generation]
    y_data = fitness
    # Create the plot
    plt.scatter(x_data, y_data, color='red', marker='o')

    # Add labels and title
    plt.xlabel("Real Value")
    plt.ylabel("Fitness")
    plt.title(f"Plot of Generation {generation_number}")

    # Display the plot
    plt.show()

#Q4.3 Report the best solution and its decoded real values.


# Run this
run_genetic_algorithm(10, 10)


# --- Unit Tests are good for you ------------------------

## test chromosomes
chromosome = Chromosome("0000", 4)
chromosome2 = Chromosome("0101", 4)

chromosome8 = Chromosome8Bit("11010101")
chromosome8_random = Chromosome8Bit(None)
chromosome16 = Chromosome16Bit("1111111111111111")

def test_get_accuracy():
    assert chromosome.get_accuracy(0, 15) == 1
    assert chromosome.get_accuracy(0, 7) == pytest.approx(.4667, rel=.001)
    assert chromosome8.get_accuracy(0, 255) == 1
    assert chromosome8.get_accuracy(-5.12, 5.12) == pytest.approx(.04015, rel=.001)

def test_get_real_value():
    assert chromosome.get_real_value(0, 15) == 0
    assert chromosome2.get_real_value(0, 15) == 5
    assert chromosome8_random.get_real_value(0, 255) > 0
    assert chromosome8.get_real_value(-2.048, 2.048) == pytest.approx(1.3733647, rel=.00001)
    assert chromosome16.get_real_value(-1.28, 1.28) == 1.28
    assert Chromosome16Bit("0011010101110101").get_real_value(-1.28, 1.28) == pytest.approx(-0.7454215304798, rel=.00000001)

def test_initialize_population():
    assert len(initialize_population(10, 8)) == 10

def test_sphere_model_fitness():
    assert chromosome16.sphere_model_fitness() == 5.12

def test_crossover():
    assert len(chromosome8.crossover(chromosome8_random, 1)) == 2

def test_mutation():
    assert chromosome16.mutation() != chromosome16.binary_string
    print(f"Parent: ${chromosome16.binary_string}, Mutated Offspring: ${chromosome16.mutation()}")