import random
import copy
import matplotlib.pyplot as plt
import numpy as np

# --- 8-Puzzle Visualization ---
def visualize_8_puzzle(board, cost, stuck=False, won=False):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, 4, 1))
    ax.set_yticks(np.arange(0, 4, 1))
    ax.grid(True, color='black', linewidth=0.7)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for i in range(3):
        for j in range(3):
            if board[i][j] != 0:
                ax.text(j + 0.5, i + 0.5, str(board[i][j]), fontsize=20, ha='center', va='center')

    title = ""
    if stuck:
        title = "STUCK! "
    elif won:
        title = "SOLVED! "
    plt.title(title + "8-Puzzle Configuration Cost: " + str(cost))
    plt.gca().invert_yaxis()
    plt.show()


# --- 8-Puzzle Algorithm ---
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

def generate_8_puzzle_instance(moves=20):
    def move_empty_tile(state_in):
        return random.choice(get_valid_neighbor_states(state_in))

    state = copy.deepcopy(goal_state)
    for _ in range(moves):
        state = move_empty_tile(state)
    return state

# Q1-1:  A heuristic function that calculates the number of misplaced tiles.
def puzzle_heuristic(state):
    # compute misplaced items
    misplaced = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != goal_state[i][j]:
                misplaced += 1
    return misplaced

#Q1-2: A function to generate all valid neighbor states from the current configuration.
def get_valid_neighbor_states(state) :
    #print("getting neighbors for state:", state)
    neighbors = []
    row, col = next((r, c) for r in range(3) for c in range(3) if state[r][c] == 0)
    possible_moves = [(row + dr, col + dc) for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]]
    valid_moves = [(r, c) for r, c in possible_moves if 0 <= r < 3 and 0 <= c < 3]
    for (new_row, new_col) in valid_moves:
        neighbor_state = copy.deepcopy(state)
        neighbor_state[row][col], neighbor_state[new_row][new_col] = neighbor_state[new_row][new_col], neighbor_state[row][col]
        neighbors.append(neighbor_state)
        #print("appended neighbor:", neighbors)
    return neighbors

#Q1-3 take a step down the hill
def take_one_step(state):
    # get the current misplaced tiles score.
    current_misplaced = puzzle_heuristic(state)
    # get any neighbor states.
    neighbors = get_valid_neighbor_states(state)
    # check each neighbor and step to it if it has a lower misplaced tile score.
    for n in neighbors :
        if n != state :
            misplaced = puzzle_heuristic(n)
            # if the misplaced tile score is greater than current,
            # throw it out, we are hill climbing.
            print("neighbor:", n, "misplaced tiles:", misplaced)
            if misplaced < current_misplaced :
                return n
    raise Exception("Stuck at local min!")

#Q1-3 take a step down the hill by checking the count of misplaced tiles for each neighbor state
# and picking one with a lower value if available. (I realized when I was answering the questions
# I had coded the steepest descent instead of a simple hill climber but figured I'd keep the code why not?
# This one is not being used
def take_one_step_steepest(state):
    # get the current misplaced tiles score.
    current_misplaced = puzzle_heuristic(state)
    # get any neighbor states.
    neighbors = get_valid_neighbor_states(state)
    neighbors_by_misplaced_score = {}
    # for each neighbor add it to a list in the dict keyed by misplace tile score.
    for n in neighbors :
        if n != state :
            misplaced = puzzle_heuristic(n)
            # if the misplaced tile score is greater than current,
            # throw it out, we are hill climbing.
            if misplaced < current_misplaced :
                # add the neighbor state with a matching or lower misplace score to the list for this score.
                neighbors_by_misplaced_score.setdefault(misplaced, []).append(n)
    print("Neighbors by Misplaced scores", neighbors_by_misplaced_score)
    if len(neighbors_by_misplaced_score) > 0 :
        # resort dict so low score is first
        sorted_neighbors = dict(sorted(neighbors_by_misplaced_score.items()))
        # pick a random state from the list of neighbor states with the lowest misplaced tile score.
        low_score = next(iter(sorted_neighbors))
        return random.choice(neighbors_by_misplaced_score.get(low_score))
    raise Exception("Stuck at local min!")

#Q1-3 - take a step down the hill by checking the count of misplaced tiles for each neighbor state
# and picking the first one with a lower value if available.
# Q1-4 - visualize the state after each step.
def hill_climbing_puzzle(state):
    print(f"Hill Climbing with Initial 8-Puzzle: {state}")
    visualize_8_puzzle(state, 0)
    i = 0
    while state != goal_state:
        try:
            state = take_one_step(state)
            i += 1
            print(f"Current State 8-Puzzle: {state} cost: {i}")
            visualize_8_puzzle(state, i, False, state == goal_state)
        except Exception as e:
            print("Error:", e)
            visualize_8_puzzle(state, i, True, state == goal_state)
            break

# --- Running and Visualizing 8-Puzzle ---

one_off = [[1, 2, 3], [4, 5, 6], [7, 0, 8]]
three_off = [[1, 5, 3], [4, 0, 6], [7, 2, 8]]
gets_stuck = [[1, 0, 5], [7, 3, 2], [8, 4, 6]]

# Q2-1 & Q2-2 Run with easy solve!
hill_climbing_puzzle([[1, 5, 2], [4, 0, 3], [7, 8, 6]])

# Q2-2 Run that gets stuck at local min
hill_climbing_puzzle(three_off)

# Run with random initial puzzle.
initial_puzzle = generate_8_puzzle_instance()
hill_climbing_puzzle(initial_puzzle)
