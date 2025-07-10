import copy
from AS3 import chromosome

############################################################################################
#                              PyTests with smol order numbers
###########################################################################################
chrom_5_random = chromosome.Chromosome(5)
chrom_5 = chromosome.Chromosome(0)
chrom_5.items.append(chromosome.Item(0, 0, 1.2))
chrom_5.items.append(chromosome.Item(1, 1, .8))
chrom_5.items.append(chromosome.Item(2, 2, .9))
chrom_5.items.append(chromosome.Item(3, 3, .2))
chrom_5.items.append(chromosome.Item(4, 4, 1.6))

chrom_3 = chromosome.Chromosome(0)
chrom_3.items.append(chromosome.Item(0, 0, 1.2))
chrom_3.items.append(chromosome.Item(1, 1, .8))
chrom_3.items.append(chromosome.Item(2, 2, .9))
chrom_3.items.append(chromosome.Item(3, 0, 1.9))
chrom_3.items.append(chromosome.Item(4, 1, .3))


def test_bin_assignments():
    assert chrom_5.bin_assignments() == [0, 1, 2, 3, 4]
    assert chrom_3.bin_assignments() == [0, 1, 2, 0, 1]


def test_get_total_weight():
    test_chrom = copy.deepcopy(chrom_5)
    assert test_chrom.get_total_weight() == 4.7
    test_chrom.items[0].weight += chromosome.MAX_BOX_WEIGHT
    assert test_chrom.get_total_weight() == 14.7


def test_get_number_of_bins_used():
    assert chrom_5.get_number_of_bins_used() == 5
    assert 0 <= chrom_5_random.get_number_of_bins_used() <= 5


def test_get_constraint_violation():
    test_chrom = copy.deepcopy(chrom_5)
    assert test_chrom.get_constraint_violation() == 0

    # put one box over weight limit.
    test_chrom.items[0].weight += chromosome.MAX_BOX_WEIGHT
    assert test_chrom.get_constraint_violation() == 1

    # put all boxes over weight limit.
    for item in test_chrom.items:
        item.weight += chromosome.MAX_BOX_WEIGHT
    assert test_chrom.get_constraint_violation() == 5


def test_get_mutation_probability():
    assert chrom_5.get_mutation_probability() == .2   # has 5 boxes!
    assert chrom_5_random.get_mutation_probability() >= .2   # will have 5 or less boxes.
    assert chromosome.Chromosome(100).get_mutation_probability() >= .01   # will have 100 or less boxes.


def test_tournament_selection():
    assert chromosome.tournament_selection(chrom_5, chrom_3) == chrom_3
    assert chromosome.tournament_selection(chrom_3, chrom_5) == chrom_3

    # chrom_3 has a box over weight limit.
    test_chrom = copy.deepcopy(chrom_3)
    test_chrom.items[0].weight += chromosome.MAX_BOX_WEIGHT
    assert chromosome.tournament_selection(chrom_5, test_chrom) == chrom_5
    assert chromosome.tournament_selection(test_chrom, chrom_5) == chrom_5

    # put all boxes over weight limit.
    for item in test_chrom.items:
        item.weight += chromosome.MAX_BOX_WEIGHT
    assert chromosome.tournament_selection(chrom_5, test_chrom) == chrom_5
    assert chromosome.tournament_selection(test_chrom, chrom_5) == chrom_5

    # put all boxes over weight limit.
    test_chrom2 = copy.deepcopy(chrom_5)
    for item in test_chrom2.items:
        item.weight += chromosome.MAX_BOX_WEIGHT
    assert chromosome.tournament_selection(test_chrom2, test_chrom) == test_chrom
    assert chromosome.tournament_selection(test_chrom, test_chrom2) == test_chrom


def test_crossover():
    test_chrom = copy.deepcopy(chrom_5)
    test_chrom2 = chromosome.Chromosome(5)
    offspring = test_chrom.crossover(test_chrom2, 1, 0)
    print("Crossover Offspring:", offspring)
    assert test_chrom.bin_assignments != chrom_5.bin_assignments
    assert offspring[0].items != test_chrom.items
    assert offspring[1].items != test_chrom.items

    offspring = test_chrom.crossover(test_chrom2, 2, 1)
    print("Crossover Offspring:", offspring)
    assert offspring[0].items != test_chrom.items
    assert offspring[1].items != test_chrom.items

    offspring = test_chrom.crossover(chrom_3, 2, 1)
    print("Crossover Offspring:", offspring)
    assert offspring[0].items != test_chrom.items
    assert offspring[1].items != test_chrom.items


def test_mutate():
    test_chrom = copy.deepcopy(chrom_5)
    test_chrom.mutate()
    assert test_chrom.bin_assignments != chrom_5.bin_assignments or test_chrom.items != chrom_5.items
    print(test_chrom)
