import pytest
from AS1 import hc_puzzel
from AS1.hc_puzzel import goal_state, generate_8_puzzle_instance, one_off, three_off, gets_stuck


# --- Unit Tests because I love them ---

def test_puzzle_heuristic():
    assert hc_puzzel.puzzle_heuristic(goal_state) == 0
    assert hc_puzzel.puzzle_heuristic(generate_8_puzzle_instance(1)) != 0
    assert hc_puzzel.puzzle_heuristic(one_off) == 2
    assert hc_puzzel.puzzle_heuristic(three_off) == 4


def test_get_valid_neighbor_states():
    neighbors = hc_puzzel.get_valid_neighbor_states(goal_state)
    assert len(neighbors) == 2   # you won, but there are 2 valid neighbor states

    neighbors = hc_puzzel.get_valid_neighbor_states(one_off)
    # print(f"one off neighbors: {neighbors}")
    assert len(neighbors) == 3

    neighbors = hc_puzzel.get_valid_neighbor_states(three_off)
    # print(f"three off neighbors: {neighbors}")
    assert len(neighbors) == 4


def test_take_one_step():
    with pytest.raises(Exception):
        hc_puzzel.take_one_step(gets_stuck)
