import math
from random import random

import numpy as np
from agents.agent import Agent
from helpers import get_valid_moves, execute_move, check_endgame, count_capture, random_move
from store import register_agent
from copy import deepcopy
import time


@register_agent("student_agent2")
class StudentAgent(Agent):
    """
    A class for your implementation. Implements Minimax with Alpha-Beta Pruning.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent2"
        self.autoplay = True
        self.weights = {"coin_parity": 25.0,
                        "mobility": 5.0,
                        "corners_captured": 30.0,
                        "stability": 25.0
                        }

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
        depth = 2  # Adjust this depth based on performance
        start_time = time.time()
        best_move = None
        time_limit = 1.985
        board_size = len(chess_board)


        while time.time() - start_time < time_limit:
          try:
            move = self.monte_carlo_tree_search(chess_board, depth, player, opponent, start_time, time_limit)
            if move is not None:
              best_move = move
            depth += 1
          except TimeoutError:
            break # stop searching if the time limit is exceeded


        time_taken = time.time() - start_time
        print(f"My AI's turn took {time_taken} seconds. Explored depth: {depth -1}")

        if best_move is not None:
            return best_move

        # Fallback to a random move if no best move found
        valid_moves = get_valid_moves(chess_board, player)
        if valid_moves:
            return valid_moves[np.random.randint(len(valid_moves))]
        return None

    def monte_carlo_tree_search(self, board, player, opponent, iterations,start_time, time_limit):
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
            if time.time() - start_time >= time_limit:
                break
            for move in valid_moves:
                new_board = deepcopy(board)
                execute_move(new_board, move, player)
                eval_value = self.evaluation_function(new_board, player, opponent)

                # Update statistics based on the heuristic evaluation
                move_stats[move]['visits'] += 1
                move_stats[move]['value'] += eval_value

            # Select the move with the highest average value
            best_move = max(valid_moves, key=lambda move: move_stats[move]['value'] / move_stats[move]['visits'])
            return best_move


    def evaluation_function(self, chess_board, player, opponent):
        """
               A simple heuristic function to evaluate the board state.

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
               int
                   The evaluation score for the board state.
               """

        coin_parity_score = self.heuristic_coin_parity(chess_board, player, opponent)
        mobility_score = self.heuristic_mobility(chess_board, player, opponent)
        corners_score = self.heuristic_corners_capture(chess_board, player, opponent)
        stability_score = self.heuristic_stability(chess_board, player, opponent)

        total_score = (
                self.weights["coin_parity"] * coin_parity_score +
                self.weights["mobility"] * mobility_score +
                self.weights["corners_captured"] * corners_score +
                self.weights["stability"] * stability_score
        )
        return total_score



        #### heuristic calculations

    def heuristic_coin_parity(self, chess_board, player, opponent):
        player_coins = np.sum(chess_board == player)
        opponent_coins = np.sum(chess_board == opponent)
        if (player_coins + opponent_coins) != 0:
            return (player_coins - opponent_coins) / (player_coins + opponent_coins)
        else:
            return 0

    def heuristic_mobility(self, chess_board, player, opponent):
        player_mobility = len(get_valid_moves(chess_board, player))
        opponent_mobility = len(get_valid_moves(chess_board, opponent))
        if (player_mobility + opponent_mobility) != 0:
            return (player_mobility - opponent_mobility) / (player_mobility + opponent_mobility)
        else:
            return 0

    def heuristic_corners_capture(self, chess_board, player, opponent):
        corners = [(0, 0),
                   (0, len(chess_board) - 1),
                   (len(chess_board) - 1, 0),
                   (len(chess_board) - 1, len(chess_board) - 1)]
        player_corners = sum(1 for x, y in corners if chess_board[x][y] == player)
        opponent_corners = sum(1 for x, y in corners if chess_board[x][y] == opponent)
        if (player_corners + opponent_corners) != 0:
            return (player_corners - opponent_corners) / (player_corners + opponent_corners)
        else:
            return 0

    def heuristic_stability(self, chess_board, player, opponent):
        board_size = len(chess_board)
        player_stability = 0
        opponent_stability = 0

        for r in range(board_size):
            for c in range(board_size):
                if chess_board[r][c] == 0:  # board space not occupied
                    continue
                elif chess_board[r][c] == player:
                    stability = self.classify_stability(chess_board, r, c, player, opponent)
                    player_stability += stability
                elif chess_board[r][c] == opponent:
                    stability = self.classify_stability(chess_board, r, c, opponent, player)
                    opponent_stability += stability

        if (player_stability + opponent_stability) != 0:
            stability_value = (player_stability - opponent_stability) / (player_stability + opponent_stability)
        else:
            stability_value = 0

        return stability_value

    def classify_stability(self, chess_board, r, c, player, opponent):
        """
        Classify the stability of a disc at position (r, c).
        Returns:
          +1 for stable discs. Will never change colors
          0 for semi-stable discs. Could potentially change in the future
          -1 for unstable discs. Could change in one move
        """
        board_size = len(chess_board)

        # Corners are always stable
        if (r, c) in [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]:
            return 1

        # check if the disk can be flanked in the next turn
        if self.is_unstable(chess_board, r, c, opponent):
            return -1

        # Discs are stable if they are fully surrounded by other stable discs
        # or are part of an edge that cannot be flipped.
        if self.is_stable(chess_board, r, c, player):
            return 1

        # otherwise, it is semi-stable
        return 0

    def is_unstable(self, chess_board, r, c, opponent):
        """
        Determine if a disc can be flanked in the next move.
        """
        board_size = len(chess_board)
        directions = [(-1, -1),
                      (-1, 0),
                      (-1, 1),
                      (0, -1),
                      (0, 1),
                      (1, -1),
                      (1, 0),
                      (1, 1)]

        for dr, dc in directions:
            x, y = r + dr, c + dc
            if 0 <= x < board_size and 0 <= y < board_size:
                if chess_board[x][y] == opponent:
                    # Check if the opponent can flank in this direction
                    nx, ny = x + dr, y + dc
                    while 0 <= nx < board_size and 0 <= ny < board_size:
                        if chess_board[nx][ny] == 0:
                            break
                        if chess_board[nx][ny] == chess_board[r][c]:
                            return True
                        nx += dr
                        ny += dc

        return False

    def is_stable(self, chess_board, r, c, player, visited=None):
        """
        Check if a disc is completely surrounded by stable discs.
        A disc is stable if all its neighbors are either:
          - Stable
          - Edges
          - Same color discs that are also stable
        """
        if visited is None:
            visited = set()

        # Prevent revisiting the same disc
        if (r, c) in visited:
            return True

        visited.add((r, c))

        board_size = len(chess_board)
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1)]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < board_size and 0 <= nc < board_size:
                # Check if neighbor is of the same color
                if chess_board[nr][nc] == player:
                    # Ensure neighbor is stable or part of an edge
                    if not self.is_edge_stable(chess_board, nr, nc, player) and \
                            not self.is_stable(chess_board, nr, nc, player, visited):
                        return False
                else:
                    # Neighbor is not the player's disc, so it's unstable
                    return False

        return True

    def is_edge_stable(self, chess_board, r, c, player):
        """
        Determine if a coin on the edge is stable. Edge coins are stable if all discs
        on the same row/column are of the same color, or bounded by stable corners.
        """
        board_size = len(chess_board)

        # Check horizontal stability
        if r == 0 or r == board_size - 1:
            if all(chess_board[r][col] == player for col in range(board_size)):
                return True

        # Check vertical stability
        if c == 0 or c == board_size - 1:
            if all(chess_board[row][c] == player for row in range(board_size)):
                return True

        return False

