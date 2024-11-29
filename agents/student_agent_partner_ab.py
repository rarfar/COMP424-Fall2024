import math
import numpy as np
from agents.agent import Agent
from helpers import get_valid_moves, execute_move, check_endgame, count_capture
from store import register_agent
from copy import deepcopy
import time


@register_agent("student_agent_partner_ab")
class StudentAgent(Agent):
    """
    A class for your implementation. Implements Minimax with Alpha-Beta Pruning.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "MinimaxAgent"
        self.autoplay = True

        # potential weights for each heuristic component
        self.weights = {"coin_parity": 1.0,
                        "mobility": 3.0,
                        "corners_captured": 5.0,
                        "stability": 4.0
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
        depth = 3  # Adjust this depth based on performance
        start_time = time.time()

        best_move = self.best_move(chess_board, depth, player, opponent)
        time_taken = time.time() - start_time
        print(f"My AI's turn took {time_taken} seconds.")

        if best_move is not None:
            return best_move

        # Fallback to a random move if no best move found
        valid_moves = get_valid_moves(chess_board, player)
        if valid_moves:
            return valid_moves[np.random.randint(len(valid_moves))]
        return None

    def minimax(self, board, depth, alpha, beta, maximizing_player, player, opponent):
        """
        Minimax algorithm with alpha-beta pruning.

        Parameters
        ----------
        board : numpy.ndarray
            The current state of the board.
        depth : int
            The depth limit for the search.
        alpha : float
            The alpha value for pruning.
        beta : float
            The beta value for pruning.
        maximizing_player : bool
            True if the current player is maximizing, False otherwise.
        player : int
            The current player (1 or 2).
        opponent : int
            The opponent player (1 or 2).

        Returns
        -------
        int
            The evaluation score for the board state.
        """
        if depth == 0 or check_endgame(board, player, opponent)[0]:
            return self.evaluation_function(board, player, opponent)

        if maximizing_player:
            max_eval = -math.inf
            for move in get_valid_moves(board, player):
                new_board = deepcopy(board)
                execute_move(new_board, move, player)
                eval = self.minimax(new_board, depth - 1, alpha, beta, False, player, opponent)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = math.inf
            for move in get_valid_moves(board, opponent):
                new_board = deepcopy(board)
                execute_move(new_board, move, opponent)
                eval = self.minimax(new_board, depth - 1, alpha, beta, True, player, opponent)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval

    def best_move(self, board, depth, player, opponent):
        """
        Find the best move using Minimax with alpha-beta pruning.

        Parameters
        ----------
        board : numpy.ndarray
            The current state of the board.
        depth : int
            The depth limit for the search.
        player : int
            The current player (1 or 2).
        opponent : int
            The opponent player (1 or 2).

        Returns
        -------
        tuple
            The (row, col) position for the best move.
        """
        best_val = -math.inf
        best_move = None
        alpha = -math.inf
        beta = math.inf
        corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0),
                   (board.shape[0] - 1, board.shape[1] - 1)]

        legal_moves = get_valid_moves(board, player)

        for move in get_valid_moves(board, player):
            # Prioritize taking corners
            for corner in corners:
                if corner in legal_moves:
                    return corner  # High-priority move
            new_board = deepcopy(board)
            execute_move(new_board, move, player)
            move_val = self.minimax(new_board, depth - 1, alpha, beta, False, player, opponent)
            if move_val > best_val:
                best_val = move_val
                best_move = move

            alpha = max(alpha, move_val)

        return best_move

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

        start_time = time.time()


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
