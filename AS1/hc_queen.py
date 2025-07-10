import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# --- 8-Queens Problem Visualization ---
def visualize_queens(board, cost, attacking_queens, stuck=False):
    chessboard = np.zeros((8, 8))
    for col, row in enumerate(board):
        chessboard[row][col] = 1

    plt.figure(figsize=(6, 6))
    sns.heatmap(chessboard, cbar=False, annot=True, square=True, linewidths=0.5, linecolor='black', cmap='Blues',
                xticklabels=False, yticklabels=False)


    title = "8-Queens Solution Cost: "
    if cost == 0:
        title = "Initial Queens Board Cost: "
    if attacking_queens == 0 :
        title = "SOLVED! " + title
    elif stuck :
        title = "STUCK! " + title
    title = title + str(cost) + " Heuristic: " + str(attacking_queens)
    plt.title(title)
    print(title, board)
    plt.show()


# --- 8-Queens Algorithm ---
# Q1-1: a function that generates a random 8 queen state
def generate_8_queens_instance():
    return [random.randint(0, 7) for _ in range(8)]


# Q1-2: A heuristic function that computes the number of pairs of queens that are attacking each other.
def queens_heuristic(board):
    attacking_pairs = 0
    # The way the data is represented, it is not possible for 2 queens to be in the same column
    # check rows
    attacking_pairs += get_attacking_pairs_from_rows(board)
    # check diagonals
    attacking_pairs += get_attacking_pairs_from_diagonals(board)
    return attacking_pairs


# Q1-2 calculate attacking pairs by row
def get_attacking_pairs_from_rows(board):
    attacking_pairs = 0
    # add up any queens on the same row,
    seen_rows = []
    for row in range(len(board)):
        # if a row number occurs twice on the board, that is 1 attacking pair,
        # 3 times is 2 attacking pairs, etc.
        row_value = board[row]
        if seen_rows.count(row_value) == 0 and board.count(row_value) > 0 :
            attacking_pairs += (board.count(board[row]) - 1)
        seen_rows.append(row_value)
    return attacking_pairs


# Q1-2 calculate attacking pairs by diagonal
def get_attacking_pairs_from_diagonals(board):
    attacking_pairs = 0
    for col in range(0, len(board)):
        attacking_pairs += check_diagonal_down_for_queens(board, col)
        attacking_pairs += check_diagonal_up_for_queens(board, col)
    return attacking_pairs


## there is probably some fancy lambda that would make this shorter, but this works and I understand it.
def check_diagonal_up_for_queens(board, current_col):
    current_row_val = board[current_col]
    # loop through the columns to the right and up checking for diagonal matches.
    for col_diff in range(1, len(board) - current_col):
        # subtract the number of columns from the row value (ie: over 2, up 2)
        diagonal_up_row_val = (current_row_val - col_diff)
        # don't bother checking off the playing surface and check
        if diagonal_up_row_val < 0 :
            return 0
        # if the value of the diagonal row matches what the board has in that column.
        if diagonal_up_row_val == board[current_col + col_diff]:
            # once we have found a match stop checking, there may be more but they are blocked from attack.
            return 1
    return 0


def check_diagonal_down_for_queens(board, current_col):
    current_row_val = board[current_col]
    # loop through the columns to the right and down checking for diagonal matches.
    for col_diff in range(1, len(board) - current_col):
        # add the number of columns to the row value, which moves down diagonally (ie over 2, down 2)
        diagonal_down_row_val = (current_row_val + col_diff)
        # don't bother checking off the playing surface and check
        if diagonal_down_row_val > 7:
            return 0
        # if the value of the diagonal row matches what the board has in that column.
        if diagonal_down_row_val == board[current_col + col_diff]:
            # once we have found a match stop checking, there may be more but they are blocked from attack.
            return 1
    return 0


#Q1-3 get a list of neighbor states moving one queen (column) at a time.
def get_possible_neighbors(board, col):
    neighbors = []
    current_row = board[col]
    for row in range(len(board)) :
        if row != current_row : # row = current row is current state.
            neighbor_state = copy.deepcopy(board)
            neighbor_state[col] = row
            neighbors.append(neighbor_state)
    return neighbors


#Q1-4 climb down the hill one step by moving one queen column by column
def take_one_step(board, col):
    #print("taking a step to a new neighbor column:", col)
    # get the current attacking queens score.
    current_attacking = queens_heuristic(board)
    # get any neighbor states.
    neighbors = get_possible_neighbors(board, col)
    # check each neighbor and step to it if it has a lower misplaced tile score.
    for n in neighbors :
        attacking_queens = queens_heuristic(n)
        # if the misplaced tile score is greater than current,
        # throw it out, we are hill climbing.
        #print("neighbor:", n, "attacking queens:", attacking_queens)
        if attacking_queens < current_attacking :
            return n
    return board


#Q1-4 Hill climbing algorithm
def hill_climbing_queens(board):
    cost = 0
    last_board = board
    last_seen_count = 0
    attacking_queens = queens_heuristic(board)
    visualize_queens(board, cost, attacking_queens)
    while attacking_queens > 0 :
        if last_seen_count > len(board):
            break
        for col in range(len(board)) :
            board = take_one_step(board, col)
            cost += 1
            attacking_queens = queens_heuristic(board)
            if board == last_board:
                # if it matches last board, increment seen.
                last_seen_count += 1
                if last_seen_count > len(board) :
                    # if it has matched 8 columns in a row, algorithm is stuck
                    break
            else :
                # if board does not match the last board, reset last board
                last_board = board
                last_seen_count = 0
    # we broke out of the loop, if attacking queens > 0, we were stuck at a local min.
    visualize_queens(board, cost, attacking_queens, attacking_queens > 0)


# --- Running and Visualizing 8-Queens ---
def main():
    # Q2-1 Run with a random initial state 5 times.
    print("run 5 random starts:")
    for i in range(5) :
        print("Run:", i + 1)
        initial_queens = generate_8_queens_instance()
        hill_climbing_queens(initial_queens)

    # double check boards from paper
    print("checking starts from assignment paper:")
    hill_climbing_queens([4, 6, 0, 1, 2, 0, 4, 1])
    hill_climbing_queens([7, 6, 7, 2, 2, 5, 3, 2])
    hill_climbing_queens([1, 0, 1, 5, 7, 7, 6, 0])
    hill_climbing_queens([4, 7, 4, 3, 2, 4, 5, 6])
    hill_climbing_queens([0, 7, 7, 7, 3, 3, 5, 7])


if __name__ == "__main__":
    main()