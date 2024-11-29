import numpy as np
import random
from agents.agent import Agent
from helpers import get_valid_moves, execute_move
from store import register_agent
from copy import deepcopy
import time


@register_agent("student_agent_partner_mc")
class SecondAgent(Agent):
    """
    A class for your implementation. Implements Monte Carlo Tree Search (MCTS).
    """

    def __init__(self):
        super(SecondAgent, self).__init__()
        self.name = "MCTSAgent"
        self.autoplay = True

    def step(self, chess_board, player, opponent):
        """
        The step function called by the simulator.

        Parameters
        ----------
        chess_board : numpy.ndarray
            The current state of the board.
        player : int
            The current player (1 or 2).
        opponent : int
            The opponent player (1 or 2).

        Returns
        -------
        tuple
            The (row, col) position for the next move.
        """
        start_time = time.time()
        iterations = 500  # Number of MCTS iterations to run

        best_move = self.monte_carlo_tree_search(chess_board, player, opponent, iterations)
        time_taken = time.time() - start_time
        print(f"My AI's turn took {time_taken} seconds.")

        if best_move is not None:
            return best_move

        # Fallback to a random move if no best move found
        valid_moves = get_valid_moves(chess_board, player)
        if valid_moves:
            return valid_moves[np.random.randint(len(valid_moves))]
        return None

    def monte_carlo_tree_search(self, board, player, opponent, iterations):
        """
        Monte Carlo Tree Search (MCTS) algorithm using a heuristic evaluation.

        Parameters
        ----------
        board : numpy.ndarray
            The current state of the board.
        player : int
            The current player (1 or 2).
        opponent : int
            The opponent player (1 or 2).
        iterations : int
            The number of iterations for the MCTS.

        Returns
        -------
        tuple
            The (row, col) position for the best move.
        """
        valid_moves = get_valid_moves(board, player)
        if not valid_moves:
            return None

        # Track the total value and number of visits for each move
        move_stats = {move: {'value': 0, 'visits': 0} for move in valid_moves}

        for _ in range(iterations):
            move = random.choice(valid_moves)
            new_board = deepcopy(board)
            execute_move(new_board, move, player)
            eval_value = self.heuristic_fn(new_board, player, opponent)

            # Update statistics based on the heuristic evaluation
            move_stats[move]['visits'] += 1
            move_stats[move]['value'] += eval_value

        # Select the move with the highest average value
        best_move = max(valid_moves, key=lambda move: move_stats[move]['value'] / move_stats[move]['visits'])
        return best_move

    def heuristic_fn(self, board, player, opponent):
        """
        A heuristic function to evaluate the board state.

        Parameters
        ----------
        board : numpy.ndarray
            The current state of the board.
        player : int
            The current player (1 or 2).
        opponent : int
            The opponent player (1 or 2).

        Returns
        -------
        int
            The evaluation score for the board state.
        """
        # Components of the heuristic: piece count, corner control, mobility
        player_discs = np.sum(board == player)
        opponent_discs = np.sum(board == opponent)
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
        player_corners = sum([1 for corner in corners if board[corner] == player])
        opponent_corners = sum([1 for corner in corners if board[corner] == opponent])
        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, opponent))

        # Weighted sum of components
        return (10 * (player_discs - opponent_discs) +
                25 * (player_corners - opponent_corners) +
                5 * (player_moves - opponent_moves))
