# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

#potential weights for each heuristic component
    self.weights = {"coin_parity": 1.0,
                      "mobility": 2.0,
                      "corners_captured":9.0,
                      "stability":8
                      }

  def heuristic_coin_parity(self, chess_board, player, opponent):
    player_coins = np.sum(chess_board == player)
    opponent_coins = np.sum(chess_board == opponent)
    if (player_coins + opponent_coins) != 0:
      return (player_coins - opponent_coins) / (player_coins + opponent_coins)
    else: return 0

  def heuristic_mobility(self, chess_board, player, opponent):
    player_mobility = len(get_valid_moves(chess_board, player))
    opponent_mobility = len(get_valid_moves(chess_board, opponent))
    if (player_mobility + opponent_mobility) != 0:
      return (player_mobility - opponent_mobility) / (player_mobility+opponent_mobility)
    else: return 0

  def heuristic_corners_capture(self, chess_board, player, opponent):
    corners = [(0,0),
               (0,len(chess_board)-1),
               (len(chess_board)-1,0),
               (len(chess_board)-1,len(chess_board)-1)]
    player_corners = sum(1 for x,y in corners if chess_board[x][y] == player)
    opponent_corners = sum(1 for x,y in corners if chess_board[x][y] == opponent)
    if (player_corners + opponent_corners) != 0:
      return (player_corners-opponent_corners) / (player_corners+opponent_corners)
    else: return 0


  def heuristic_stability(self, chess_board, player, opponent):

    board_size = len(chess_board)
    player_stability = 0
    opponent_stability = 0

    for r in range(board_size):
      for c in range(board_size):
        if chess_board[r][c] == 0: # board space not occupied
          continue
        elif chess_board[r][c] == player:
          stability = self.classify_stability(chess_board, r, c, player, opponent)
          player_stability += stability
        elif chess_board[r][c] == opponent:
          stability = self.classify_stability(chess_board, r, c, opponent, player)
          opponent_stability += stability

    if (player_stability + opponent_stability) !=0:
      stability_value = (player_stability - opponent_stability)/(player_stability + opponent_stability)
    else: stability_value = 0

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

    #Corners are always stable
    if (r, c) in [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]:
      return 1

    #check if the disk can be flanked in the next turn
    if self.is_unstable(chess_board, r, c, opponent):
      return -1

    #Discs are stable if they are fully surrounded by other stable discs
    # or are part of an edge that cannot be flipped.
    if self.is_stable(chess_board, r, c, player):
      return 1

    #otherwise, it is semi-stable
    return 0

  def is_unstable(self,chess_board, r, c, opponent):
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


  def evaluation_function(self, chess_board, player, opponent):
    #Combine all heurisitic components with weights
    coin_parity_score = self.heuristic_coin_parity(chess_board,player,opponent)
    mobility_score = self.heuristic_mobility(chess_board, player, opponent)
    corners_score = self.heuristic_corners_capture(chess_board, player, opponent)

    #STABILITY STUCK IN RECURSIVE LOOP, comment out for now
    stability_score = self.heuristic_stability(chess_board, player, opponent)

    total_score = (
            self.weights["coin_parity"] * coin_parity_score +
            self.weights["mobility"] * mobility_score +
            self.weights["corners_captured"] * corners_score +
            self.weights["stability"] * stability_score
    )
    return total_score


  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Get all legal moves for the current player (our student agent)
    legal_moves = get_valid_moves(chess_board, player)

    if not legal_moves:
      return None  # No valid moves available, pass turn

    best_move = None
    best_score = -float("inf")
    for move in legal_moves:
      # Create a copy of the board and simulate the move
      simulated_board = deepcopy(chess_board)
      execute_move(simulated_board, move, player)

      # Evaluate the resulting board state
      score = self.evaluation_function(simulated_board, player, opponent)
      if score > best_score:
        best_score = score
        best_move = move

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    time_taken = time.time() - start_time

    #print("My AI's turn took ", time_taken, "seconds.")


    return best_move



