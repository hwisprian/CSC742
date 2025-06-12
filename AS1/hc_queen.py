import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pytest

# --- 8-Queens Problem Visualization ---
def visualize_queens(board, cost):
    chessboard = np.zeros((8, 8))
    for col, row in enumerate(board):
        chessboard[row][col] = 1

    plt.figure(figsize=(6, 6))
    sns.heatmap(chessboard, cbar=False, annot=True, square=True, linewidths=0.5, linecolor='black', cmap='Blues',
                xticklabels=False, yticklabels=False)
    plt.title("8-Queens Solution Cost: "+str(cost))
    plt.show()


# --- 8-Queens Algorithm ---
# Q1: a function that generates a random 8 queen state
def generate_8_queens_instance():
    return [random.randint(0, 7) for _ in range(8)]

# Q2: A heuristic function that computes the number of pairs of queens that are attacking each other.
def queens_heuristic(board):
    attacking_pairs = 0
    # The way the data is represented, it is not possible for 2 queens to be in the same column
    # check rows
    attacking_pairs += get_attacking_pairs_from_rows(board)
    # check diagonals
    attacking_pairs += get_attacking_pairs_from_diagonals(board)
    return attacking_pairs

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

def get_attacking_pairs_from_diagonals(board):
    attacking_pairs = 0
    for col in range(0, len(board)):
        attacking_pairs += check_diagonal_down_for_queens(board, col)
        attacking_pairs += check_diagonal_up_for_queens(board, col)
    return attacking_pairs


def check_diagonal_up_for_queens(board, current_col):
    current_row_val = board[current_col]
    # loop through the columns to the right and up checking for diagonal matches.
    for col in range(current_col + 1, len(board) - current_col):
        # subtract the number of columns from the row value
        diagonal_up_row_val = (current_row_val - col)
        # don't bother checking off the playing surface and check
        if diagonal_up_row_val < 0 :
            return 0
        # if the value of the diagonal row matches what the board has in the column.
        if diagonal_up_row_val >= 0 and diagonal_up_row_val == board[col]:
            # once we have found a match stop checking, there may be more but they are blocked from attack.
            return 1
    return 0


def check_diagonal_down_for_queens(board, current_col):
    current_row_val = board[current_col]
    # loop through the columns to the right and down checking for diagonal matches.
    for col in range(current_col + 1, len(board) - current_col):
        # add the number of columns to the row value, which moves down diagonally.
        diagonal_down_row_val = (current_row_val + col)
        # don't bother checking off the playing surface and check
        if diagonal_down_row_val > 7:
            return 0
        # if the value of the diagonal row matches what the board has in the column.
        if diagonal_down_row_val == board[col]:
            # once we have found a match stop checking, there may be more but they are blocked from attack.
            return 1
    return 0

def hill_climbing_queens(board):
    pass


# --- Running and Visualizing 8-Queens ---
initial_queens = generate_8_queens_instance()
print(f"Initial Queens Board: {initial_queens}")
visualize_queens(initial_queens, 0)


# --- Unit Tests ---------------------------------

test_board_seq=[0, 1, 2, 3, 4, 5, 6, 7]
test_board_dsc=[7, 6, 5, 4, 3, 2, 1, 0]
test_board_1 = [1, 1, 1, 1, 1, 1, 1, 1]
test_board =   [3, 5, 2, 7, 2, 4, 1, 0]

def test_get_attacking_pairs_from_rows():
    assert get_attacking_pairs_from_rows(test_board_seq) == 0
    assert get_attacking_pairs_from_rows(test_board_dsc) == 0
    assert get_attacking_pairs_from_rows(test_board_1) == 7
    assert get_attacking_pairs_from_rows(test_board) == 1

def test_check_diagonal_up_for_queens():
    # up from zero would be -1 off the playing surface
    assert check_diagonal_up_for_queens(test_board_seq, 0) == 0
    assert check_diagonal_up_for_queens(test_board_seq, 1) == 0

def test_check_diagonal_down_for_queens():
    # down from zero would be 1 which is a match on the sequential board.
    assert check_diagonal_down_for_queens(test_board_seq, 0) == 1
    assert check_diagonal_down_for_queens(test_board_seq, 1) == 1

def test_get_attacking_pairs_from_diagonals():
    assert get_attacking_pairs_from_diagonals(test_board_seq) == 7
    assert get_attacking_pairs_from_diagonals(test_board_dsc) == 7
    assert get_attacking_pairs_from_diagonals(test_board_1) == 0
    assert get_attacking_pairs_from_diagonals(test_board) == 3 # (5,7), (5,2), (1,0)
