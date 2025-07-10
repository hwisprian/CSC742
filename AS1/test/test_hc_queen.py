from AS1 import hc_queen

# --- Unit Tsts ---------------------------------

board_seq = [0, 1, 2, 3, 4, 5, 6, 7]
board_dsc = [7, 6, 5, 4, 3, 2, 1, 0]
board_1 = [1, 1, 1, 1, 1, 1, 1, 1]
board_2 = [3, 5, 2, 7, 2, 4, 1, 0]


def test_get_attacking_pairs_from_rows():
    assert hc_queen.get_attacking_pairs_from_rows(board_seq) == 0
    assert hc_queen.get_attacking_pairs_from_rows(board_dsc) == 0
    assert hc_queen.get_attacking_pairs_from_rows(board_1) == 7
    assert hc_queen.get_attacking_pairs_from_rows(board_2) == 1


def test_check_diagonal_up_for_queens():
    # up from zero would be -1 off the playing surface
    assert hc_queen.check_diagonal_up_for_queens(board_seq, 0) == 0
    assert hc_queen.check_diagonal_up_for_queens(board_seq, 1) == 0
    assert hc_queen.check_diagonal_up_for_queens(board_dsc, 0) == 1
    assert hc_queen.check_diagonal_up_for_queens(board_dsc, 1) == 1


def test_check_diagonal_down_for_queens():
    # down from zero would be 1 which is a match on the sequential board.
    assert hc_queen.check_diagonal_down_for_queens(board_seq, 0) == 1
    assert hc_queen.check_diagonal_down_for_queens(board_seq, 1) == 1
    assert hc_queen.check_diagonal_down_for_queens(board_dsc, 0) == 0
    assert hc_queen.check_diagonal_down_for_queens(board_dsc, 1) == 0


def test_get_attacking_pairs_from_diagonals():
    assert hc_queen.get_attacking_pairs_from_diagonals(board_seq) == 7
    assert hc_queen.get_attacking_pairs_from_diagonals(board_dsc) == 7
    assert hc_queen.get_attacking_pairs_from_diagonals(board_1) == 0
    assert hc_queen.get_attacking_pairs_from_diagonals(board_2) == 3  # (5,7), (5,2), (1,0)
