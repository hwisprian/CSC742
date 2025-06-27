import copy
import math
import random
import pytest

############################################################################################
#          Q1 - a chromosome that has a list of binary strings
#               upon creation if no binary strings are provided it will generate as needed
#               for the given length and number of variables for the objective functions
###############################################################################################
class Chromosome:
    binary_strings = []
    binary_string_length = 8 # default to 8 why not
    decoded_real_values = []
    generation = 0
    fitness = 0

    # Q1 Constructor, takes binary strings list or generate a list random binary string
    # of size number of variables  if one is not provided.
    def __init__(self, strings_in, string_length=8, number_of_variables=1, gen_num=0):
        ## init my saved attributes
        self.binary_strings = []
        self.binary_string_length = string_length
        self.decoded_real_values = []
        self.generation = gen_num
        self.fitness = 0
        # if no binary strings were passed in generate some of the length specified.
        if strings_in is None:
            # if no binary strings are provided generate a random ones
            for _ in range(number_of_variables):
                binary_string = ''
                for _ in range(self.binary_string_length):
                    binary_string += str(random.randint(0,1))
                self.binary_strings.append(binary_string)
        else:
            # binary strings were provided, so make sure we got the correct length on there.
            self.binary_strings = strings_in
            self.binary_string_length = len(strings_in[0])

    def __str__(self):
        string = 'Chromosome ('
        for i in range(len(self.binary_strings)):
            if i > 0:
                string += ", "
            if i > 0 and i % 2 == 0:
                string += "\n"
            string += "x" + str(i+1) + "=" + self.binary_strings[i]
        string += ")\n gen: " + str(self.generation) + " fitness: " + str(self.fitness)
        return string

    # precision given the upper and lower bound and binary string length we are decoding.
    def get_precision(self, lower_bound, upper_bound):
        return (upper_bound - lower_bound) / (pow(2, self.binary_string_length) - 1)

    # Q2.1 decode binary chromosomes into real numbers based on provided lower and upper bound
    def get_real_values(self, lower_bound, upper_bound):
        precision = self.get_precision(lower_bound, upper_bound)
        real_values = []
        for i in range(len(self.binary_strings)):
            binary_string = self.binary_strings[i]
            real_values.append( lower_bound + (precision * int(binary_string, 2)) )
        self.decoded_real_values = real_values # Q4.2 save these for later we want to display them :)
        return real_values

    #############################################################################################
    # Q2.2 implement evaluation functions for all the De Jong functions
    ##############################################################################################

    # Q2.2 implement de john sphere model function
    def sphere_model_fitness(self):
        self.fitness = 0
        x = self.get_real_values(-5.12, 5.12)
        for i in range(len(x)):
            xi = x[i]**2
            self.fitness += xi
        return self.fitness

    # Q2.2 implement De Jong weighted sphere function
    def weighted_sphere_model_fitness(self):
        x = self.get_real_values(-2.048, 2.048)
        # we may have more than 2 variables, but we only care about the first 2.
        self.fitness = 100 * ( x[0]**2 - x[1] )**2  + (1-x[0])**2
        return self.fitness

    # Q2.2 implement De Jong step function
    def step_function_fitness(self):
        self.fitness = 0
        x = self.get_real_values(-5.12, 5.12)
        for i in range(len(x)):
            xi = math.floor(x[i])
            self.fitness += xi
        return self.fitness


    # Q2.2 implement De Jong noisy quartic function
    def noisy_quartic_fitness(self):
        self.fitness = 0
        x = self.get_real_values(-1.28, 1.28)
        for i in range(len(x)):
            xi = x[i]
            self.fitness += (i * xi**4 + random.random())
        return self.fitness

#################################################################################################
# Q3 - Genetic Algorithm Operations - Crossover and Mutation
######################################################################################################
    # Q3.2 one or two point cross over
    def crossover(self, parent2, gen_num, number_of_points, probability):
        num_strings = len(self.binary_strings)
        offspring1_strings = [''] * num_strings
        offspring2_strings = [''] * num_strings

        if random.random() < probability:
            crossover_points = []
            for i in range(number_of_points):
                crossover_points.append(random.randint(1, self.binary_string_length))
            crossover_points.sort()
            for j in range(len(self.binary_strings)):
                for i in range(self.binary_string_length):
                    if (i < crossover_points[0] or
                            (len(crossover_points) > 1 and  i >= crossover_points[1])):
                        offspring1_strings[j] += (self.binary_strings[j][i])
                        offspring2_strings[j] += (parent2.binary_strings[j][i])
                    else:
                        offspring1_strings[j] += (parent2.binary_strings[j][i])
                        offspring2_strings[j] += (self.binary_strings[j][i])

        else:
            offspring1_strings = copy.deepcopy(parent2.binary_strings)
            offspring2_strings = copy.deepcopy(self.binary_strings)
        return [Chromosome(offspring1_strings, gen_num=gen_num), Chromosome(offspring2_strings, gen_num=gen_num)]

    #Q3.3 bitwise mutation
    def mutation(self):
        mutant = copy.deepcopy(self)
        for i in range(len(self.binary_strings)):
            mutated_string = ''
            for bit in mutant.binary_strings[i]:
                new_bit = bit
                # mutate the bits with a probability of 1 / total length of bit string
                if random.random() < 1 / (self.binary_string_length * len(self.binary_strings)):
                    new_bit = '1' if bit == '0' else '1' ## flip a bit
                mutated_string += new_bit
            mutant.binary_strings[i] = mutated_string
        return mutant

    def get_decimal_values(self):
        decimals = []
        for binary_string in self.binary_strings :
            return int(binary_string, 2)
        return decimals

##################################################################################################
# --- Unit Tests are good for you ------------------------
#################################################################################################
## test chromosomes
chromosome = Chromosome(["0000"], string_length=4)
chromosome2 = Chromosome(["0101"], string_length=4)

chromosome8 = Chromosome(["11010101"])
chromosome8_random = Chromosome(None)
chromosome16_zero= Chromosome(["0000000000000000", "0000000000000000"], string_length=16)
chromosome16_one = Chromosome(["1111111111111111", "1111111111111111"], string_length=16)

def test_str():
    assert chromosome8.__str__() == "Chromosome (x1=11010101) gen: 0 fit: 0"
    assert chromosome16_zero.__str__() == "Chromosome (x1=0000000000000000, x2=0000000000000000) gen: 0 fit: 0"
    assert chromosome16_one.__str__() == "Chromosome (x1=1111111111111111, x2=1111111111111111) gen: 0 fit: 0"

def test_get_precision():
    assert chromosome.get_precision(0, 15) == 1
    assert chromosome.get_precision(0, 7) == pytest.approx(.4667, rel=.001)
    assert chromosome8.get_precision(0, 255) == 1
    assert chromosome8.get_precision(-5.12, 5.12) == pytest.approx(.04015, rel=.001)

def test_get_real_value():
    assert chromosome.get_real_values(0, 15)[0] == 0
    assert chromosome2.get_real_values(0, 15)[0] == 5
    assert chromosome8_random.get_real_values(0, 255)[0] > 0
    assert chromosome8.get_real_values(-2.048, 2.048)[0] == pytest.approx(1.3733647, rel=.00001)
    assert chromosome16_zero.get_real_values(-2.048, 2.048) == -2.048
    assert chromosome16_one.get_real_values(-1.28, 1.28)[0] == 1.28
    assert Chromosome(["0011010101110101"], string_length=16).get_real_values(-1.28, 1.28)[0] == pytest.approx(-0.7454215304798, rel=.00000001)

def test_sphere_model_fitness():
    assert chromosome16_zero.sphere_model_fitness() == -5.12 ** 2 + -5.12 ** 2
    assert chromosome16_one.sphere_model_fitness() == 5.12 ** 2 + 5.12 ** 2

def test_weighted_sphere_model_fitness():
    assert chromosome16_zero.weighted_sphere_model_fitness() == 100 * ( (-2.048)**2 - -2.048)**2 + (1 - -2.048)**2
    assert chromosome16_one.weighted_sphere_model_fitness() == 100 * ( 2.048**2 - 2.048)**2 + (1 - 2.048)**2

def test_step_function_fitness():
    assert chromosome16_zero.step_function_fitness() == math.floor(-5.12) + math.floor(-5.12)
    assert chromosome16_one.step_function_fitness() == math.floor(5.12) + math.floor(5.12)

def test_noisy_quartic_fitness():
    assert chromosome16_zero.step_function_fitness() == -12
    assert chromosome16_one.step_function_fitness() == 10

def test_crossover():
    children = chromosome16_zero.crossover(chromosome16_one, 1, 2,.9)
    print(f"Original: ${chromosome16_zero.binary_strings} , ${chromosome16_one.binary_strings}, "
          f"Resulting Children: ${children[0].binary_strings} ${children[1].binary_strings}")

    # we got 2 children back.
    assert len(children) == 2
    # the binary strings are the right length.
    assert len(children[0].binary_strings[0]) == 16
    assert len(children[0].binary_strings[1]) == 16

    children = chromosome16_zero.crossover(chromosome16_one, 2, 1,.9)
    print(f"Original: ${chromosome16_zero.binary_strings} , ${chromosome16_one.binary_strings}, "
          f"Resulting Children: ${children[0].binary_strings} ${children[1].binary_strings}")
    # we got 2 children back.
    assert len(children) == 2
    # the binary strings are the right length.
    assert len(children[0].binary_strings[0]) == 16
    assert len(children[0].binary_strings[1]) == 16

def test_mutation():
    mutated_chromosome = chromosome16_one.mutation()
    # this could fail, if a bit flips in mutation.
    assert mutated_chromosome.binary_strings == chromosome16_one.binary_strings
    print(f"Original: ${chromosome16_one.binary_strings}, Mutated: ${mutated_chromosome.binary_strings}")
