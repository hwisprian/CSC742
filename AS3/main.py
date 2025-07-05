from chromosome import *

if __name__ == '__main__':
    # Q 4.1
    run_genetic_algorithm(50, 20, 10, 1)
    run_genetic_algorithm(50, 20, 25, 1)
    run_genetic_algorithm(50, 20, 50, 1)
    run_genetic_algorithm(50, 20, 100, 1)

    run_genetic_algorithm(50, 20, 10, 2)
    run_genetic_algorithm(50, 20, 25, 2)
    run_genetic_algorithm(50, 20, 50, 2)
    run_genetic_algorithm(50, 20, 100, 2)

    # Q 4.2 vary the population size and max generations
    run_genetic_algorithm(50, 50, 10, 2)
    run_genetic_algorithm(50, 50, 25, 2)
    run_genetic_algorithm(50, 50, 50, 2)
    run_genetic_algorithm(50, 50, 100, 2)

    run_genetic_algorithm(20, 50, 10, 2)
    run_genetic_algorithm(20, 50, 25, 2)
    run_genetic_algorithm(20, 50, 50, 2)
    run_genetic_algorithm(20, 50, 100, 2)

    run_genetic_algorithm(100, 20, 100, 2)
    run_genetic_algorithm(100, 20, 1000, 2)