"""
Tic-Tac-Toe Q-Learning Game (3x3 Board)
========================================
Uses Q-Learning (reinforcement learning) to train an AI agent to play Tic-Tac-Toe.
The agent learns through self-play, then plays against a human player via a Pygame UI.

Features:
- 3x3 board with 3-in-a-row win condition
- Pink & White fresh UI design
- Welcome screen with game rules
- 10-round match system
- Real-time Q-value display
"""

import random
import pickle
import os
import sys

import pygame

# ============================================================================
# Game Configuration
# ============================================================================
GRID_SIZE = 3       # 3x3 board
WIN_LENGTH = 3      # 3 in a row to win
TOTAL_CELLS = GRID_SIZE * GRID_SIZE  # 9 cells

# ============================================================================
# 1. Game Environment
# ============================================================================

class TicTacToeEnv:
    """
    Tic-Tac-Toe game environment (4x4 board, 3-in-a-row to win).

    State space: Each cell can be 0 (empty), 1 (X), or 2 (O).
                 Total possible states: 3^16 (not all reachable).
                 State is represented as a tuple of 16 integers.

    Action space: 0-15, representing the 16 cells on the board.
                  Valid actions are cells that are currently empty.

    Reward function:
        +1.0  for winning
        -1.0  for losing
        +0.5  for a draw
        -0.05 for each move (encourages faster wins)
    """

    def __init__(self):
        self.winning_lines = self._generate_winning_lines()
        self.reset()

    def _generate_winning_lines(self):
        """Generate all possible winning lines (3-in-a-row) on a 4x4 board."""
        lines = []
        
        # Horizontal lines
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE - WIN_LENGTH + 1):
                line = tuple(row * GRID_SIZE + col + i for i in range(WIN_LENGTH))
                lines.append(line)
        
        # Vertical lines
        for col in range(GRID_SIZE):
            for row in range(GRID_SIZE - WIN_LENGTH + 1):
                line = tuple((row + i) * GRID_SIZE + col for i in range(WIN_LENGTH))
                lines.append(line)
        
        # Diagonal lines (top-left to bottom-right)
        for row in range(GRID_SIZE - WIN_LENGTH + 1):
            for col in range(GRID_SIZE - WIN_LENGTH + 1):
                line = tuple((row + i) * GRID_SIZE + col + i for i in range(WIN_LENGTH))
                lines.append(line)
        
        # Diagonal lines (top-right to bottom-left)
        for row in range(GRID_SIZE - WIN_LENGTH + 1):
            for col in range(WIN_LENGTH - 1, GRID_SIZE):
                line = tuple((row + i) * GRID_SIZE + col - i for i in range(WIN_LENGTH))
                lines.append(line)
        
        return lines

    def reset(self):
        """Reset the board to initial empty state."""
        self.board = [0] * TOTAL_CELLS  # 0=empty, 1=X, 2=O
        self.current_player = 1  # X goes first
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        """Return current state as a tuple (hashable for Q-table)."""
        return tuple(self.board)

    def get_valid_actions(self):
        """Return list of valid actions (empty cells)."""
        return [i for i in range(TOTAL_CELLS) if self.board[i] == 0]

    def step(self, action):
        """
        Execute an action for the current player.

        Returns: (next_state, reward, done, info)
        """
        if self.done:
            raise ValueError("Game is already over")
        if self.board[action] != 0:
            raise ValueError(f"Cell {action} is already occupied")

        self.board[action] = self.current_player

        # Check for winner
        winner = self._check_winner()
        if winner is not None:
            self.done = True
            self.winner = winner
            if winner == self.current_player:
                reward = 1.0   # Win
            else:
                reward = -1.0  # Loss (shouldn't happen in normal flow)
            return self.get_state(), reward, True, {"winner": winner}

        # Check for draw
        if len(self.get_valid_actions()) == 0:
            self.done = True
            self.winner = 0  # Draw
            return self.get_state(), 0.5, True, {"winner": 0}

        # Game continues - small negative reward to encourage faster wins
        reward = -0.05
        # Switch player
        self.current_player = 3 - self.current_player  # Toggle between 1 and 2
        return self.get_state(), reward, False, {}

    def _check_winner(self):
        """Check if there's a winner. Returns player number or None."""
        b = self.board
        for line in self.winning_lines:
            cells = [b[i] for i in line]
            if cells[0] != 0 and all(c == cells[0] for c in cells):
                return cells[0]
        return None

    def get_winning_line(self):
        """Get the cells that form the winning line."""
        b = self.board
        for line in self.winning_lines:
            cells = [b[i] for i in line]
            if cells[0] != 0 and all(c == cells[0] for c in cells):
                return line
        return None


# ============================================================================
# 2. Q-Learning Agent
# ============================================================================

class QLearningAgent:
    """
    Q-Learning agent for Tic-Tac-Toe.

    Uses a Q-table (dictionary) mapping (state, action) -> Q-value.
    Implements epsilon-greedy exploration strategy.
    """

    def __init__(self, player_id, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01):
        self.player_id = player_id
        self.lr = learning_rate          # Alpha: how much to update Q-values
        self.gamma = discount_factor     # Gamma: importance of future rewards
        self.epsilon = epsilon           # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}                # Q-table: {(state, action): q_value}

    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair, default 0."""
        return self.q_table.get((state, action), 0.0)

    def get_all_q_values(self, state, valid_actions):
        """Get Q-values for all valid actions in a state."""
        return {a: self.get_q_value(state, a) for a in valid_actions}

    def choose_action(self, state, valid_actions, training=True):
        """
        Choose an action using epsilon-greedy strategy.

        During training: explore with probability epsilon, exploit otherwise.
        During play: always exploit (choose best action).
        """
        if not valid_actions:
            return None

        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(valid_actions)
        else:
            # Exploitation: choose action with highest Q-value
            q_values = {a: self.get_q_value(state, a) for a in valid_actions}
            max_q = max(q_values.values())
            # Break ties randomly
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_valid_actions, done):
        """
        Update Q-value using the Q-Learning formula:
        Q(s,a) = Q(s,a) + lr * (reward + gamma * max(Q(s',a')) - Q(s,a))
        """
        current_q = self.get_q_value(state, action)

        if done or not next_valid_actions:
            target = reward
        else:
            max_next_q = max(self.get_q_value(next_state, a) for a in next_valid_actions)
            target = reward + self.gamma * max_next_q

        # Q-Learning update
        new_q = current_q + self.lr * (target - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save Q-table to file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filepath):
        """Load Q-table from file."""
        with open(filepath, "rb") as f:
            self.q_table = pickle.load(f)


# ============================================================================
# 3. Training
# ============================================================================

def train(episodes=200000):
    """
    Train two Q-Learning agents through self-play on 4x4 board.

    Both agents learn simultaneously. After training, the agent for player 1 (X)
    is saved and used to play against the human.
    """
    env = TicTacToeEnv()

    agent_x = QLearningAgent(player_id=1, learning_rate=0.15, discount_factor=0.95,
                             epsilon=1.0, epsilon_decay=0.99997, epsilon_min=0.05)
    agent_o = QLearningAgent(player_id=2, learning_rate=0.15, discount_factor=0.95,
                             epsilon=1.0, epsilon_decay=0.99997, epsilon_min=0.05)

    agents = {1: agent_x, 2: agent_o}
    stats = {"X_wins": 0, "O_wins": 0, "draws": 0}

    for episode in range(episodes):
        state = env.reset()
        # Store history of (state, action, agent) for delayed reward update
        history = []

        while not env.done:
            current_agent = agents[env.current_player]
            valid_actions = env.get_valid_actions()
            action = current_agent.choose_action(state, valid_actions, training=True)

            old_state = state
            state, reward, done, info = env.step(action)

            history.append((old_state, action, current_agent, env.current_player))

            if done:
                # Update all agents based on game outcome
                winner = info.get("winner", 0)
                for h_state, h_action, h_agent, h_next_player in history:
                    if winner == 0:
                        # Draw
                        final_reward = 0.5
                    elif winner == h_agent.player_id:
                        final_reward = 1.0
                    else:
                        final_reward = -1.0
                    h_agent.update(h_state, h_action, final_reward, state, [], True)

                # Track stats
                if winner == 1:
                    stats["X_wins"] += 1
                elif winner == 2:
                    stats["O_wins"] += 1
                else:
                    stats["draws"] += 1

        # Decay epsilon for both agents
        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

        # Print progress every 10000 episodes
        if (episode + 1) % 10000 == 0:
            total = stats["X_wins"] + stats["O_wins"] + stats["draws"]
            print(f"Episode {episode + 1}/{episodes} | "
                  f"X wins: {stats['X_wins']/total*100:.1f}% | "
                  f"O wins: {stats['O_wins']/total*100:.1f}% | "
                  f"Draws: {stats['draws']/total*100:.1f}% | "
                  f"Epsilon: {agent_x.epsilon:.4f} | "
                  f"Q-table size: {len(agent_x.q_table)}")

    print(f"\nTraining complete! Final Q-table sizes: X={len(agent_x.q_table)}, O={len(agent_o.q_table)}")

    # Save both agents
    script_dir = os.path.dirname(os.path.abspath(__file__))
    agent_x.save(os.path.join(script_dir, "agent_x.pkl"))
    agent_o.save(os.path.join(script_dir, "agent_o.pkl"))
    print("Agents saved.")

    return agent_x, agent_o


# ============================================================================
# 4. Pygame UI - Black & White Minimalist Design
# ============================================================================

# Colors - Pink & White Fresh Theme
WHITE = (255, 255, 255)
BLACK = (40, 40, 40)
PINK = (255, 182, 193)           # Light pink for X
PINK_DARK = (255, 105, 140)      # Darker pink for accents
PINK_LIGHT = (255, 240, 245)     # Very light pink background
GRAY_LIGHT = (245, 245, 245)
GRAY_MID = (200, 200, 200)
GRAY_DARK = (120, 120, 120)
PINK_HOVER = (255, 220, 230)     # Hover highlight
PINK_BTN = (255, 150, 170)       # Button color
PINK_BTN_HOVER = (255, 120, 150) # Button hover

# Layout - 3x3 Board
CELL_SIZE = 100
BOARD_PIXEL_SIZE = CELL_SIZE * GRID_SIZE  # 300 pixels
PADDING = 30
INFO_PANEL_WIDTH = 250
WINDOW_WIDTH = PADDING + BOARD_PIXEL_SIZE + PADDING + INFO_PANEL_WIDTH + PADDING
WINDOW_HEIGHT = 500
BOARD_X = PADDING
BOARD_Y = 80

# Game states
STATE_WELCOME = 0
STATE_PLAYING = 1
STATE_GAME_OVER = 2
STATE_MATCH_END = 3

MAX_ROUNDS = 10


class TicTacToeGame:
    """Pygame-based Tic-Tac-Toe game with Q-Learning AI opponent."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe | Q-Learning AI")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("Arial", 42, bold=True)
        self.font_subtitle = pygame.font.SysFont("Arial", 20)
        self.font_large = pygame.font.SysFont("Arial", 64, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 18)
        self.font_small = pygame.font.SysFont("Arial", 14)
        self.font_button = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_header = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_xo = pygame.font.SysFont("Arial", 48, bold=True)

        # Game state
        self.game_state = STATE_WELCOME
        self.env = TicTacToeEnv()
        self.agent = QLearningAgent(player_id=2)  # AI plays as O

        # Load trained agent
        script_dir = os.path.dirname(os.path.abspath(__file__))
        agent_path = os.path.join(script_dir, "agent_o.pkl")
        if os.path.exists(agent_path):
            self.agent.load(agent_path)
            self.agent.epsilon = 0  # No exploration during play
            print(f"Loaded trained agent (Q-table size: {len(self.agent.q_table)})")
        else:
            print("No trained agent found. Training now...")
            _, agent_o = train()
            self.agent = agent_o
            self.agent.epsilon = 0

        # Match state
        self.current_round = 0
        self.player_score = 0
        self.ai_score = 0
        self.draws = 0

        # UI state
        self.hover_cell = -1
        self.winning_line = None
        self.last_result = ""
        self.current_q_values = {}
        self.last_ai_action = None
        self.last_ai_q_value = None

    def reset_match(self):
        """Reset for a new 10-round match."""
        self.current_round = 0
        self.player_score = 0
        self.ai_score = 0
        self.draws = 0
        self.reset_round()
        self.game_state = STATE_PLAYING

    def reset_round(self):
        """Reset for a new round."""
        self.env.reset()
        self.winning_line = None
        self.last_result = ""
        self.current_q_values = {}
        self.last_ai_action = None
        self.last_ai_q_value = None
        self.update_q_values()

    def update_q_values(self):
        """Update current Q-values for display."""
        state = self.env.get_state()
        valid_actions = self.env.get_valid_actions()
        self.current_q_values = self.agent.get_all_q_values(state, valid_actions)

    def get_cell_from_mouse(self, pos):
        """Convert mouse position to board cell index."""
        mx, my = pos
        if BOARD_X <= mx < BOARD_X + BOARD_PIXEL_SIZE and BOARD_Y <= my < BOARD_Y + BOARD_PIXEL_SIZE:
            col = (mx - BOARD_X) // CELL_SIZE
            row = (my - BOARD_Y) // CELL_SIZE
            return row * GRID_SIZE + col
        return -1

    # ========== Drawing Methods ==========

    def draw_welcome_screen(self):
        """Draw the welcome/start screen."""
        self.screen.fill(PINK_LIGHT)

        # Title
        title = self.font_title.render("TIC-TAC-TOE", True, PINK_DARK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 60))
        self.screen.blit(title, title_rect)

        subtitle = self.font_subtitle.render("Q-Learning AI", True, GRAY_DARK)
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 100))
        self.screen.blit(subtitle, subtitle_rect)

        # Decorative board preview (4x4)
        preview_size = 140
        preview_x = (WINDOW_WIDTH - preview_size) // 2
        preview_y = 125
        cell = preview_size // GRID_SIZE

        pygame.draw.rect(self.screen, GRAY_MID, (preview_x, preview_y, preview_size, preview_size), 2)
        for i in range(1, GRID_SIZE):
            pygame.draw.line(self.screen, GRAY_MID,
                             (preview_x + i * cell, preview_y),
                             (preview_x + i * cell, preview_y + preview_size), 1)
            pygame.draw.line(self.screen, GRAY_MID,
                             (preview_x, preview_y + i * cell),
                             (preview_x + preview_size, preview_y + i * cell), 1)

        # Draw sample X and O
        font_preview = pygame.font.SysFont("Arial", 28, bold=True)
        x_text = font_preview.render("X", True, PINK_DARK)
        o_text = font_preview.render("O", True, BLACK)
        self.screen.blit(x_text, (preview_x + cell // 2 - 8, preview_y + cell // 2 - 12))
        self.screen.blit(o_text, (preview_x + cell * 2 + cell // 2 - 8, preview_y + cell + cell // 2 - 12))

        # Rules section
        rules_y = 290
        rules_title = self.font_header.render("GAME RULES", True, BLACK)
        rules_rect = rules_title.get_rect(center=(WINDOW_WIDTH // 2, rules_y))
        self.screen.blit(rules_title, rules_rect)

        # Underline
        pygame.draw.line(self.screen, PINK_DARK,
                         (WINDOW_WIDTH // 2 - 50, rules_y + 15),
                         (WINDOW_WIDTH // 2 + 50, rules_y + 15), 2)

        rules = [
            "You play as X (first move)",
            "AI plays as O (Q-Learning)",
            "Get 3 in a row to win",
            "Complete 10 rounds to finish",
            "Highest score wins!"
        ]

        for i, rule in enumerate(rules):
            text = self.font_small.render(f"{i+1}. {rule}", True, GRAY_DARK)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, rules_y + 40 + i * 22))
            self.screen.blit(text, text_rect)

        # Start button
        self.draw_button("START GAME", WINDOW_WIDTH // 2, 470, "start")

    def draw_button(self, text, cx, cy, btn_id):
        """Draw a button and store its rect."""
        btn_w, btn_h = 160, 45
        btn_rect = pygame.Rect(cx - btn_w // 2, cy - btn_h // 2, btn_w, btn_h)

        mx, my = pygame.mouse.get_pos()
        is_hover = btn_rect.collidepoint(mx, my)

        if is_hover:
            pygame.draw.rect(self.screen, PINK_BTN_HOVER, btn_rect, border_radius=8)
            text_surf = self.font_button.render(text, True, WHITE)
        else:
            pygame.draw.rect(self.screen, PINK_BTN, btn_rect, border_radius=8)
            text_surf = self.font_button.render(text, True, WHITE)

        text_rect = text_surf.get_rect(center=btn_rect.center)
        self.screen.blit(text_surf, text_rect)

        setattr(self, f"btn_{btn_id}", btn_rect)

    def draw_playing_screen(self):
        """Draw the main game screen."""
        self.screen.fill(PINK_LIGHT)

        # Header
        round_text = self.font_header.render(f"ROUND {self.current_round + 1} / {MAX_ROUNDS}", True, PINK_DARK)
        self.screen.blit(round_text, (PADDING, 20))

        # Turn indicator
        if not self.env.done:
            if self.env.current_player == 1:
                turn = "Your turn (X)"
            else:
                turn = "AI thinking..."
        else:
            turn = self.last_result
        turn_text = self.font_medium.render(turn, True, GRAY_DARK)
        self.screen.blit(turn_text, (PADDING, 45))

        # Draw board
        self.draw_board()

        # Draw info panel
        self.draw_info_panel()

    def draw_board(self):
        """Draw the game board."""
        # Board background
        pygame.draw.rect(self.screen, WHITE,
                         (BOARD_X, BOARD_Y, BOARD_PIXEL_SIZE, BOARD_PIXEL_SIZE))
        pygame.draw.rect(self.screen, GRAY_MID,
                         (BOARD_X, BOARD_Y, BOARD_PIXEL_SIZE, BOARD_PIXEL_SIZE), 2)

        # Grid lines
        for i in range(1, GRID_SIZE):
            # Vertical
            x = BOARD_X + i * CELL_SIZE
            pygame.draw.line(self.screen, GRAY_MID, (x, BOARD_Y), (x, BOARD_Y + BOARD_PIXEL_SIZE), 2)
            # Horizontal
            y = BOARD_Y + i * CELL_SIZE
            pygame.draw.line(self.screen, GRAY_MID, (BOARD_X, y), (BOARD_X + BOARD_PIXEL_SIZE, y), 2)

        # Hover highlight
        if self.hover_cell >= 0 and not self.env.done:
            if self.env.board[self.hover_cell] == 0 and self.env.current_player == 1:
                row, col = self.hover_cell // GRID_SIZE, self.hover_cell % GRID_SIZE
                hover_rect = pygame.Rect(BOARD_X + col * CELL_SIZE + 2,
                                         BOARD_Y + row * CELL_SIZE + 2,
                                         CELL_SIZE - 4, CELL_SIZE - 4)
                pygame.draw.rect(self.screen, PINK_HOVER, hover_rect)

        # Draw X's and O's
        for i in range(TOTAL_CELLS):
            row, col = i // GRID_SIZE, i % GRID_SIZE
            cx = BOARD_X + col * CELL_SIZE + CELL_SIZE // 2
            cy = BOARD_Y + row * CELL_SIZE + CELL_SIZE // 2

            if self.env.board[i] == 1:  # X - Pink color
                offset = CELL_SIZE // 4
                pygame.draw.line(self.screen, PINK_DARK,
                                 (cx - offset, cy - offset),
                                 (cx + offset, cy + offset), 4)
                pygame.draw.line(self.screen, PINK_DARK,
                                 (cx + offset, cy - offset),
                                 (cx - offset, cy + offset), 4)
            elif self.env.board[i] == 2:  # O - Black color
                pygame.draw.circle(self.screen, BLACK, (cx, cy), CELL_SIZE // 4, 3)

        # Winning line highlight
        if self.winning_line:
            for cell in self.winning_line:
                row, col = cell // GRID_SIZE, cell % GRID_SIZE
                rect = pygame.Rect(BOARD_X + col * CELL_SIZE + 4,
                                   BOARD_Y + row * CELL_SIZE + 4,
                                   CELL_SIZE - 8, CELL_SIZE - 8)
                s = pygame.Surface((CELL_SIZE - 8, CELL_SIZE - 8), pygame.SRCALPHA)
                s.fill((0, 0, 0, 40))
                self.screen.blit(s, rect.topleft)

    def draw_info_panel(self):
        """Draw the information panel on the right side."""
        panel_x = BOARD_X + BOARD_PIXEL_SIZE + PADDING
        panel_y = BOARD_Y

        # === SCORE ===
        self.draw_section_header("SCORE", panel_x, panel_y)
        panel_y += 25

        score_text = f"You (X): {self.player_score}    AI (O): {self.ai_score}    Draw: {self.draws}"
        st = self.font_small.render(score_text, True, BLACK)
        self.screen.blit(st, (panel_x, panel_y))
        panel_y += 35

        # === AI ACTION ===
        self.draw_section_header("AI LAST ACTION", panel_x, panel_y)
        panel_y += 25

        if self.last_ai_action is not None:
            r, c = self.last_ai_action // 3, self.last_ai_action % 3
            action_text = f"Position: ({r}, {c})"
            q_text = f"Q-value: {self.last_ai_q_value:.4f}" if self.last_ai_q_value else "Q-value: N/A"
        else:
            action_text = "Waiting..."
            q_text = ""

        at = self.font_small.render(action_text, True, BLACK)
        self.screen.blit(at, (panel_x, panel_y))
        panel_y += 18

        if q_text:
            qt = self.font_small.render(q_text, True, GRAY_DARK)
            self.screen.blit(qt, (panel_x, panel_y))
        panel_y += 30

        # === Q-VALUES ===
        self.draw_section_header("Q-VALUES (AI View)", panel_x, panel_y)
        panel_y += 25

        # Draw mini Q-value grid (3x3)
        mini_cell = 50
        mini_x = panel_x
        mini_y = panel_y

        pygame.draw.rect(self.screen, GRAY_MID,
                         (mini_x, mini_y, mini_cell * GRID_SIZE, mini_cell * GRID_SIZE), 1)
        for i in range(1, GRID_SIZE):
            pygame.draw.line(self.screen, GRAY_MID,
                             (mini_x + i * mini_cell, mini_y),
                             (mini_x + i * mini_cell, mini_y + mini_cell * GRID_SIZE), 1)
            pygame.draw.line(self.screen, GRAY_MID,
                             (mini_x, mini_y + i * mini_cell),
                             (mini_x + mini_cell * GRID_SIZE, mini_y + i * mini_cell), 1)

        # Fill Q-values
        for i in range(TOTAL_CELLS):
            row, col = i // GRID_SIZE, i % GRID_SIZE
            cx = mini_x + col * mini_cell + mini_cell // 2
            cy = mini_y + row * mini_cell + mini_cell // 2

            if self.env.board[i] == 1:
                text = self.font_small.render("X", True, PINK_DARK)
            elif self.env.board[i] == 2:
                text = self.font_small.render("O", True, BLACK)
            elif i in self.current_q_values:
                q_val = self.current_q_values[i]
                text = self.font_small.render(f"{q_val:.1f}", True, GRAY_DARK)
            else:
                text = self.font_small.render("-", True, GRAY_MID)

            text_rect = text.get_rect(center=(cx, cy))
            self.screen.blit(text, text_rect)

        panel_y += mini_cell * GRID_SIZE + 15

        # === Q-LEARNING INFO ===
        self.draw_section_header("Q-LEARNING INFO", panel_x, panel_y)
        panel_y += 25

        info_lines = [
            f"Q-table size: {len(self.agent.q_table)}",
            f"Learning rate: 0.1",
            f"Discount factor: 0.95",
            f"Strategy: Greedy"
        ]

        for line in info_lines:
            text = self.font_small.render(line, True, GRAY_DARK)
            self.screen.blit(text, (panel_x, panel_y))
            panel_y += 18

    def draw_section_header(self, text, x, y):
        """Draw a section header with underline."""
        header = self.font_header.render(text, True, BLACK)
        self.screen.blit(header, (x, y))
        pygame.draw.line(self.screen, PINK, (x, y + 18), (x + 120, y + 18), 2)

    def draw_game_over_screen(self):
        """Draw the game over screen for a single round."""
        # Draw the playing screen as background
        self.draw_playing_screen()

        # Overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((255, 240, 245, 220))
        self.screen.blit(overlay, (0, 0))

        # Game Over text
        go_text = self.font_title.render("GAME OVER", True, PINK_DARK)
        go_rect = go_text.get_rect(center=(WINDOW_WIDTH // 2, 180))
        self.screen.blit(go_text, go_rect)

        # Result
        result_text = self.font_subtitle.render(self.last_result, True, BLACK)
        result_rect = result_text.get_rect(center=(WINDOW_WIDTH // 2, 230))
        self.screen.blit(result_text, result_rect)

        # Score
        score = f"Score - You: {self.player_score}  AI: {self.ai_score}  Draw: {self.draws}"
        score_text = self.font_medium.render(score, True, GRAY_DARK)
        score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, 280))
        self.screen.blit(score_text, score_rect)

        # Rounds remaining
        remaining = MAX_ROUNDS - self.current_round
        if remaining > 0:
            remain_text = self.font_small.render(f"{remaining} rounds remaining", True, GRAY_DARK)
            remain_rect = remain_text.get_rect(center=(WINDOW_WIDTH // 2, 320))
            self.screen.blit(remain_text, remain_rect)
            self.draw_button("NEXT ROUND", WINDOW_WIDTH // 2, 380, "next")
        else:
            self.game_state = STATE_MATCH_END

    def draw_match_end_screen(self):
        """Draw the final match results screen."""
        self.screen.fill(PINK_LIGHT)

        # Title
        title = self.font_title.render("MATCH COMPLETE", True, PINK_DARK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 80))
        self.screen.blit(title, title_rect)

        # Determine winner
        if self.player_score > self.ai_score:
            winner = "YOU WIN!"
            winner_color = PINK_DARK
        elif self.ai_score > self.player_score:
            winner = "AI WINS!"
            winner_color = BLACK
        else:
            winner = "IT'S A TIE!"
            winner_color = GRAY_DARK

        winner_text = self.font_large.render(winner, True, winner_color)
        winner_rect = winner_text.get_rect(center=(WINDOW_WIDTH // 2, 160))
        self.screen.blit(winner_text, winner_rect)

        # Final score board
        pygame.draw.rect(self.screen, WHITE,
                         (WINDOW_WIDTH // 2 - 150, 210, 300, 120))
        pygame.draw.rect(self.screen, PINK,
                         (WINDOW_WIDTH // 2 - 150, 210, 300, 120), 2)

        score_title = self.font_header.render("FINAL SCORE", True, BLACK)
        score_rect = score_title.get_rect(center=(WINDOW_WIDTH // 2, 230))
        self.screen.blit(score_title, score_rect)

        you_text = self.font_medium.render(f"You (X):  {self.player_score}", True, PINK_DARK)
        you_rect = you_text.get_rect(center=(WINDOW_WIDTH // 2, 265))
        self.screen.blit(you_text, you_rect)

        ai_text = self.font_medium.render(f"AI (O):   {self.ai_score}", True, BLACK)
        ai_rect = ai_text.get_rect(center=(WINDOW_WIDTH // 2, 290))
        self.screen.blit(ai_text, ai_rect)

        draw_text = self.font_medium.render(f"Draws:    {self.draws}", True, GRAY_DARK)
        draw_rect = draw_text.get_rect(center=(WINDOW_WIDTH // 2, 315))
        self.screen.blit(draw_text, draw_rect)

        # Stats
        total = self.player_score + self.ai_score + self.draws
        stats = f"Total rounds: {total} | Win rate: {self.player_score/total*100:.0f}%"
        stats_text = self.font_small.render(stats, True, GRAY_DARK)
        stats_rect = stats_text.get_rect(center=(WINDOW_WIDTH // 2, 370))
        self.screen.blit(stats_text, stats_rect)

        # Play again button
        self.draw_button("PLAY AGAIN", WINDOW_WIDTH // 2, 440, "again")

    # ========== Game Logic ==========

    def ai_move(self):
        """Let the AI agent make a move."""
        if self.env.done or self.env.current_player != 2:
            return

        state = self.env.get_state()
        valid_actions = self.env.get_valid_actions()
        action = self.agent.choose_action(state, valid_actions, training=False)

        if action is not None:
            self.last_ai_q_value = self.agent.get_q_value(state, action)
            self.last_ai_action = action

            _, reward, done, info = self.env.step(action)
            self.update_q_values()

            if done:
                self.handle_round_end(info)

    def handle_round_end(self, info):
        """Handle the end of a round."""
        winner = info.get("winner", 0)
        self.winning_line = self.env.get_winning_line()
        self.current_round += 1

        if winner == 1:
            self.player_score += 1
            self.last_result = "You win this round!"
        elif winner == 2:
            self.ai_score += 1
            self.last_result = "AI wins this round!"
        else:
            self.draws += 1
            self.last_result = "It's a draw!"

        self.game_state = STATE_GAME_OVER

    def handle_click(self, pos):
        """Handle mouse click based on current game state."""
        if self.game_state == STATE_WELCOME:
            if hasattr(self, "btn_start") and self.btn_start.collidepoint(pos):
                self.reset_match()

        elif self.game_state == STATE_PLAYING:
            if self.env.done or self.env.current_player != 1:
                return

            cell = self.get_cell_from_mouse(pos)
            if cell >= 0 and self.env.board[cell] == 0:
                _, reward, done, info = self.env.step(cell)
                self.update_q_values()

                if done:
                    self.handle_round_end(info)
                else:
                    # AI's turn
                    self.draw_playing_screen()
                    pygame.display.flip()
                    pygame.time.wait(300)
                    self.ai_move()

        elif self.game_state == STATE_GAME_OVER:
            if hasattr(self, "btn_next") and self.btn_next.collidepoint(pos):
                self.reset_round()
                self.game_state = STATE_PLAYING

        elif self.game_state == STATE_MATCH_END:
            if hasattr(self, "btn_again") and self.btn_again.collidepoint(pos):
                self.game_state = STATE_WELCOME

    def run(self):
        """Main game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.hover_cell = self.get_cell_from_mouse(event.pos)

            # Draw based on state
            if self.game_state == STATE_WELCOME:
                self.draw_welcome_screen()
            elif self.game_state == STATE_PLAYING:
                self.draw_playing_screen()
            elif self.game_state == STATE_GAME_OVER:
                self.draw_game_over_screen()
            elif self.game_state == STATE_MATCH_END:
                self.draw_match_end_screen()

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


# ============================================================================
# 5. Main Entry Point
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    agent_file = os.path.join(script_dir, "agent_o.pkl")

    if "--train" in sys.argv or not os.path.exists(agent_file):
        print("=" * 60)
        print("Training Q-Learning agents through self-play...")
        print("=" * 60)
        train()
        print()

    print("Starting Tic-Tac-Toe game with Q-Learning AI...")
    print("Close the window to quit.\n")

    game = TicTacToeGame()
    game.run()


if __name__ == "__main__":
    main()

