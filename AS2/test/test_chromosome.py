import pytest
from AS2 import main
import math
from AS2.chromosome import Chromosome
from AS2.main import ConvergenceException

##################################################################################################
# --- Unit Tests are good for you ------------------------
#################################################################################################


def test_initialize_population():
    init_pop = main.initialize_population(10, 8, 4)
    assert len(init_pop) == 10
    assert len(init_pop[0].binary_strings) == 4
    assert len(init_pop[0].binary_strings[0]) == 8


def test_selection():
    mating_pool = main.selection([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
    assert len(mating_pool) == 6
    print("Mating Pool:", mating_pool)

    mating_pool = main.selection([1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6])
    assert len(mating_pool) == 6
    print("Mating Pool:", mating_pool)

    with pytest.raises(ConvergenceException, match="Convergence"):
        main.selection([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


##################################################################################################
# --- ChromosomeS!  Unit Tests are good for you ------------------------
#################################################################################################
# test chromosomes
chromosome = Chromosome(["0000"], string_length=4)
chromosome2 = Chromosome(["0101"], string_length=4)

chromosome8 = Chromosome(["11010101"])
chromosome8_random = Chromosome(None)
chromosome16_zero = Chromosome(["0000000000000000", "0000000000000000"], string_length=16)
chromosome16_one = Chromosome(["1111111111111111", "1111111111111111"], string_length=16)


def test_str():
    assert chromosome8.__str__() == "Chromosome (x1=11010101)\n gen: 0 fitness: 0"
    assert chromosome16_zero.__str__() == "Chromosome (x1=0000000000000000, x2=0000000000000000)\n gen: 0 fitness: 0"
    assert chromosome16_one.__str__() == "Chromosome (x1=1111111111111111, x2=1111111111111111)\n gen: 0 fitness: 0"


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
    assert chromosome16_zero.get_real_values(-2.048, 2.048) == [-2.048, -2.048]
    assert chromosome16_one.get_real_values(-1.28, 1.28)[0] == 1.28
    assert (Chromosome(["0011010101110101"], string_length=16).get_real_values(-1.28, 1.28)[0] ==
            pytest.approx(-0.7454215304798, rel=.00000001))


def test_sphere_model_fitness():
    assert chromosome16_zero.sphere_model_fitness() == -(-(5.12 ** 2) + -(5.12 ** 2))
    assert chromosome16_one.sphere_model_fitness() == 5.12 ** 2 + 5.12 ** 2


def test_weighted_sphere_model_fitness():
    assert chromosome16_zero.weighted_sphere_model_fitness() == 100 * ((-2.048)**2 - -2.048)**2 + (1 - -2.048)**2
    assert chromosome16_one.weighted_sphere_model_fitness() == 100 * (2.048**2 - 2.048)**2 + (1 - 2.048)**2


def test_step_function_fitness():
    assert chromosome16_zero.step_function_fitness() == math.floor(-5.12) + math.floor(-5.12)
    assert chromosome16_one.step_function_fitness() == math.floor(5.12) + math.floor(5.12)


def test_noisy_quartic_fitness():
    assert chromosome16_zero.step_function_fitness() == -12
    assert chromosome16_one.step_function_fitness() == 10


def test_crossover():
    children = chromosome16_zero.crossover(chromosome16_one, 1, 2, .9)
    print(f"Original: ${chromosome16_zero.binary_strings} , ${chromosome16_one.binary_strings}, "
          f"Resulting Children: ${children[0].binary_strings} ${children[1].binary_strings}")

    # we got 2 children back.
    assert len(children) == 2
    # the binary strings are the right length.
    assert len(children[0].binary_strings[0]) == 16
    assert len(children[0].binary_strings[1]) == 16

    children = chromosome16_zero.crossover(chromosome16_one, 2, 1, .9)
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
