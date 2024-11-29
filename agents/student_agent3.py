# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy, copy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent3")
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
                      "corners_captured":4.0,
                      "stability":3
                      }


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


  def minmax(self, chess_board, player, opponent, depth, alpha, beta):
      corners = [(0, 0), (0, chess_board.shape[1] - 1), (chess_board.shape[0] - 1, 0), (chess_board.shape[0] - 1, chess_board.shape[1] - 1)]
      new_board = deepcopy(chess_board)
      legal_moves = get_valid_moves(chess_board, player)

     ###prioritize taking corners over anything else
      for corner in corners:
          for move in legal_moves:
              if move == corner:
                  return move, 0
      #base case
      if depth == 0 or len(legal_moves) == 0:
          best_move, best_score = None, self.evaluation_function(chess_board, player, opponent)
          return best_move , best_score


      if player == 1:
          best_score = -float("inf")
          best_move = None

          for move in legal_moves:
              execute_move(new_board, move, player)

              _, value =  self.minmax(new_board, 3-player, 3-opponent, depth - 1, alpha, beta)

              if value > best_score:
                  best_score = value
                  best_move = move
              alpha = max(alpha, best_score)
              if beta <= alpha:
                  break
              #regen board, dont want past simulated moves on next computation
              new_board = deepcopy(chess_board)

      if player == 2:
          best_score = float("inf")
          best_move = None

          for move in legal_moves:
              execute_move(new_board, move, player)

              _, value = self.minmax(new_board,3- player, 3- opponent, depth - 1, alpha, beta)

              if value < best_score:
                  best_score = value
                  best_move = move
              beta = min(beta, best_score)
              if beta <= alpha:
                  break

              new_board = deepcopy(chess_board)


  def iterative_deepening_search(self, chess_board, player, opponent, max_time):
      best_move = None
      depth = 1
      start_time = time.time()

      while True:
          if time.time() - start_time >= max_time:
              break
          best_move_at_depth, score = self.minmax2(chess_board,player,opponent,depth, -float('inf'), float('inf'))
          if time.time() - start_time >= max_time:
              break
          best_move = best_move_at_depth
          depth += 1

      return best_move



  def step(self, chess_board, player, opponent):
      max_time = 2.00
      move = self.iterative_deepening_search(chess_board, player, opponent, max_time)
      return move


  def heuristic_coin_parity(self, chess_board, player, opponent):
    player_coins = np.sum(chess_board == player)
    opponent_coins = np.sum(chess_board == opponent)

    return player_coins - opponent_coins

  def heuristic_mobility(self, chess_board, player, opponent):
    player_mobility = len(get_valid_moves(chess_board, player))
    opponent_mobility = len(get_valid_moves(chess_board, opponent))

    return player_mobility - opponent_mobility

  def heuristic_corners_capture(self, chess_board, player, opponent):
    corners = [(0,0),
               (0,len(chess_board)-1),
               (len(chess_board)-1,0),
               (len(chess_board)-1,len(chess_board)-1)]
    player_corners = sum(1 for x,y in corners if chess_board[x][y] == player)
    opponent_corners = sum(1 for x,y in corners if chess_board[x][y] == opponent)

    return player_corners - opponent_corners


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

    return player_stability - opponent_stability

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


  def minmax2(self, chess_board, player, opponent, depth, alpha, beta):

    corners = [(0, 0), (0, chess_board.shape[1] - 1), (chess_board.shape[0] - 1, 0),
               (chess_board.shape[0] - 1, chess_board.shape[1] - 1)]
    legal_moves = get_valid_moves(chess_board, player)

    # Prioritize taking corners
    for corner in corners:
        if corner in legal_moves:
            return corner, 0  # High-priority move

    # Base case: evaluate if no moves or depth limit
    if depth == 0 or not legal_moves:
        best_score = self.evaluation_function(chess_board, player, opponent)
        return None, best_score

    best_move = None

    if player == 1:  # Maximizing player
        best_score = -float("inf")
        for move in legal_moves:
            new_board = deepcopy(chess_board)
            execute_move(new_board, move, player)

            _, value = self.minmax2(new_board, 3 - player, 3 - opponent, depth - 1, alpha, beta)

            if value > best_score:
                best_score = value
                best_move = move

            alpha = max(alpha, best_score)
            if beta <= alpha:  # Prune
                break

    else:  # Minimizing player
        best_score = float("inf")
        for move in legal_moves:
            new_board = deepcopy(chess_board)
            execute_move(new_board, move, player)

            _, value = self.minmax2(new_board, 3 - player, 3 - opponent, depth - 1, alpha, beta)

            if value < best_score:
                best_score = value
                best_move = move

            beta = min(beta, best_score)
            if beta <= alpha:  # Prune
                break

    return best_move, best_score
