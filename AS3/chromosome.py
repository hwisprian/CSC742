import copy
import random
import matplotlib.pyplot as plt
from numpy.ma.extras import average

MAX_BOX_WEIGHT = 10.0  # in kg
MAX_ITEM_WEIGHT = 2.0  # in kg
MIN_ITEM_WEIGHT = 0.1  # avoid exact 0 for realism


############################################################################################################
#                                   Problem domain
##########################################################################################################
# An item with a weight and a box assignment
class Item:
    def __init__(self, id_in, box_id_in, weight_in):
        # create a new box assign the id and an item.
        self.item_id = id_in
        self.box_id = box_id_in
        self.weight = weight_in

    def __repr__(self):
        return f"Item(box_id={self.box_id}, weight={self.weight}) \n"


# a chromosome that has a list of items that know what box they are packed in.
class Chromosome:
    # takes a list of item_weights and / or order count.
    # if item_weights is none it will generate random items.
    def __init__(self, num_items, gen_num=0):
        self.items = []
        if num_items:
            item_weights = generate_random_item_weights(num_items)
            self.randomly_assign_items_to_bins(item_weights)
        self.generation = gen_num

    def __repr__(self):
        return (f"Chromosome(bins_used={self.get_number_of_bins_used()}, total_weight={self.get_total_weight()}, "
                f"generation={self.generation}, bin_assignments={self.bin_assignments()})")

    # Q 1.4 a function to generate a random population
    def randomly_assign_items_to_bins(self, item_weights):
        # loop through the items and put them in a box.
        # worst case we will get 1 item per box
        item_id = 0
        for weight in item_weights:
            bin_id = random.randint(0, len(item_weights) - 1)
            item = Item(item_id, bin_id, weight)
            self.items.append(item)
            item_id += 1

    # Q 3.4.3 ensure gene only holds integer values....
    def bin_assignments(self):
        return [item.box_id for item in self.items]

    # Q2.1 a function that knows how many boxes we used by counting unique ids in the list of boxes.
    def get_number_of_bins_used(self):
        distinct_box_ids = set(item.box_id for item in self.items)
        return len(distinct_box_ids)

    def get_total_weight(self):
        return round(sum(item.weight for item in self.items), 1)

    # Q2.1 fitness is number of bins used
    def fitness(self):
        # simply using number of bins used quickly goes down to zero boxes.
        return self.get_number_of_bins_used()

    # Q2.2 check for constraint violation, number of bins whose weight is > 10kg
    #   if no box is over the weight limit g = 0 (feasible)
    #   if one box is over the weight limit g = 1
    #   if all bins exceed the limit g = N
    def get_constraint_violation(self):
        violation_count = 0
        bins = self.sum_weights_by_box_id()
        for box_weight in bins.values():
            if box_weight > MAX_BOX_WEIGHT:
                violation_count += 1

        if violation_count == 0:
            return 0
        if violation_count < self.get_number_of_bins_used():
            # print("constraint violation", violation_count, "boxes over 10kg")
            return 1
        return self.get_number_of_bins_used()

    def sum_weights_by_box_id(self):
        totals = {}
        for item in self.items:
            if item.box_id in totals:
                totals[item.box_id] += item.weight
            else:
                totals[item.box_id] = item.weight
        return totals

    # Q3.3 one or two point crossover.
    def crossover(self, other_parent, number_of_points, gen_number):
        # create offspring.
        offspring_1 = Chromosome(0, gen_number)
        offspring_2 = Chromosome(0, gen_number)

        # print("crossing over:",  all_items)

        # create random crossover points that exist on both chromosomes
        crossover_points = sorted(random.sample(range(len(self.items)-1), number_of_points))
        # print("crossoverpoints:", crossover_points)
        for i in range(len(self.items)):
            item1 = copy.deepcopy(self.items[i])
            item2 = copy.deepcopy(other_parent.items[i])

            if (i < crossover_points[0] or
                    (len(crossover_points) > 1 and i >= crossover_points[1])):
                # copy the gene as is.
                offspring_1.items.append(item1)
                offspring_2.items.append(item2)
            else:
                # copy the gene from the other parent.
                offspring_1.items.append(item2)
                offspring_2.items.append(item1)
        return [offspring_1, offspring_2]

    # Q3.4.1 mutation probability.
    def get_mutation_probability(self):
        return 1 / self.get_number_of_bins_used()

    # Q 3.4
    def mutate(self):
        for i in range(len(self.items)):
            # Q 3.4.1 for each gene mutate it w/ probably 1/number of boxes.
            if random.random() < self.get_mutation_probability():
                # Q 3.4.2 assign the item to a new random bin.
                item = self.items[i]
                new_box_id = random.randint(0, len(self.items))
                if item.box_id != new_box_id:
                    item.box_id = new_box_id


# Q 1.4 Function to generate random initial population of items by their weights
# their weight being a random number between min and max weights
def generate_random_item_weights(num_items):
    return [round(random.uniform(MIN_ITEM_WEIGHT, MAX_ITEM_WEIGHT), 1) for _ in range(num_items)]


# Q1.4 Function to create initial population
def generate_initial_population(pop_size, num_items):
    population = []
    for _ in range(pop_size):
        population.append(Chromosome(num_items))
    return population


################################################################################################
#                            Q 3 Genetic Algorithm.
##############################################################################################
# Q 3.1 Tournament Selection
def tournament_selection(parent_1, parent_2):
    # Q3.2 if both are feasible / constraint violation matches
    if parent_1.get_constraint_violation() == parent_2.get_constraint_violation():
        # return the chromosome with the best fitness.
        if parent_1.fitness() < parent_2.fitness():
            return parent_1
        else:
            return parent_2

    # Q3.2 if one is feasible and the other is not, return the feasible answer.
    if parent_1.get_constraint_violation() < parent_2.get_constraint_violation():
        return parent_1
    return parent_2

########################################################################################################################
#                        Genetic Algorithm Over various generations and populations...
#######################################################################################################################


# Q 4.1 Run the genetic algorithm for 50 generations with a population of 20
# with a variable number of orders [10, 25, 50, 100]
def run_genetic_algorithm(number_generations=50, population=20, number_orders=10, crossover_points=1):
    generation = generate_initial_population(population, number_orders)
    generation_fitness = [chrom.fitness() for chrom in generation if chrom.get_constraint_violation() == 0]

    # hold on to the best solution.
    prev_best_chromosome = min(generation, key=lambda chrom: chrom.fitness() if chrom.get_constraint_violation() == 0 else float('inf'))

    # plot points is an array lists of points to plot
    #   first list is the best fitness for the generation
    #   second list is the average fitness for the generation
    plot_points = [[], []]

    for i in range(number_generations):
        generation = reproduce(generation, crossover_points, i)
        # Q 4.3 plot the best and average fitness
        # only consider solutions that do not violate constraints!!!
        generation_fitness = [chrom.fitness() for chrom in generation if chrom.get_constraint_violation() == 0]
        try:
            best_chromosome = min(
                generation,
                key=lambda chrom: chrom.fitness() if chrom.get_constraint_violation() == 0 else float('inf')
            )
            plot_points[0].append(best_chromosome.fitness())
            plot_points[1].append(average(generation_fitness))

            # keep track if we have a new best!
            if best_chromosome.fitness() < prev_best_chromosome.fitness():
                prev_best_chromosome = best_chromosome
        except ValueError:
            print("convergence! everything is zero")
            break

    plot_fitness(population, number_generations, prev_best_chromosome, number_orders, crossover_points, plot_points)


def reproduce(population, crossover_points, gen_number):
    next_generation = []
    while len(next_generation) < len(population):
        mating_pool = build_mating_pool(population)
        parent_1 = random.choice(mating_pool)
        parent_2 = random.choice(mating_pool)
        offspring = parent_1.crossover(parent_2, crossover_points, gen_number)
        if offspring:
            offspring[0].mutate()
            offspring[1].mutate()
            next_generation.extend(offspring)
    return next_generation


def build_mating_pool(population):
    mating_pool = []
    while len(mating_pool) < len(population):
        # Q 3.1 choose 2 individuals at random.
        parent_1 = random.choice(population)
        parent_2 = random.choice(population)
        # Q 3.2 determine a winner via tournament selection.
        mating_pool.append(tournament_selection(parent_1, parent_2))
    return mating_pool


# Q4.3 Plot best fitness and average fitness per generation for each function
def plot_fitness(population, number_generations, best_chromosome, number_orders, crossover_points, plot_points):
    xdata = list(range(len(plot_points[0])))
    # Create the plot
    plt.plot(xdata, plot_points[0], color='red', label='best fitness', marker='+')
    plt.plot(xdata, plot_points[1], color='green',  label='avg fitness', marker='.')
    plt.scatter(best_chromosome.generation, best_chromosome.fitness(), color='blue', label='best solution', marker='x')

    plt.legend(loc='best', fancybox=True, shadow=True, fontsize='small')

    # Add labels and title
    plt.xlabel("Generation")
    plt.ylabel("Fitness (y)")
    plt.title(f"Plot of Pop size {population} for {number_generations} "
              f"Generations for {number_orders} items and {crossover_points} crossover points")
    print(f"Best Solution: {best_chromosome}")

    # Display the plot
    plt.show()
