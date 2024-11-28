# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent2")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent2"

#potential weights for each heuristic component
    self.weights = {"coin_parity": 1.0,
                      "mobility": 3.0,
                      "corners_captured":5.0,
                      "stability":4.0
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

  def alphabeta(self,chess_board,player, opponent, depth,alpha,beta,maximizing_player):

      #base case
      if depth == 0 or check_endgame(chess_board,player, opponent):
          return self.evaluation_function(chess_board, player, opponent)

      if maximizing_player:
          max_score = -float("inf")
          legal_moves_player = get_valid_moves(chess_board, player)
          for move in legal_moves_player:
              simulated_board = deepcopy(chess_board)
              execute_move(simulated_board, move, player)
              eval = self.alphabeta(simulated_board, player, opponent, depth-1,alpha,beta,False)
              max_score = max(max_score,eval)
              alpha = max(alpha,eval)
              if beta<= alpha:
                  break #beta cutoff
          return max_score
      else:
          min_score = float("inf")
          legal_moves_opponent = get_valid_moves(chess_board, opponent)
          for move in legal_moves_opponent:
              simulated_board = deepcopy(chess_board)
              execute_move(simulated_board, move, opponent)
              eval = self.alphabeta(simulated_board, opponent, player, depth -1, alpha, beta, True)
              min_score = min(min_score,eval)
              beta = min(beta,eval)
              if beta<=alpha:
                  break
          return min_score


  def iterative_deepening_search(self,chess_board,player,opponent,max_time):
    best_move = random_move(chess_board,player) # placeholder until something better
    depth = 1
    start_time = time.time()

    while True:
        best_move_at_depth = self.alphabeta_search(chess_board,player,opponent, depth)
        if time.time() - start_time >= max_time:
            break

        best_move = best_move_at_depth
        depth += 1

    return best_move

  def alphabeta_search(self,chess_board,player,opponent,depth):
      legal_moves = get_valid_moves(chess_board, player)
      best_move = random_move(chess_board,player)
      best_value = -float("inf")

      for move in legal_moves:
          new_board = deepcopy(chess_board)
          execute_move(new_board, move, player)
          move_value = self.alphabeta(new_board,player, opponent,depth-1,-float("inf"),float("inf"), False)
          if move_value > best_value:
              best_value = move_value
              best_move = move
      return best_move


  def step(self, chess_board, player, opponent):
    max_time = 2.00
    best_move = self.iterative_deepening_search(chess_board,player,opponent,max_time)
    return best_move

  #### heuristic calculations
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

