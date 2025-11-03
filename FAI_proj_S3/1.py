# -*- coding: utf-8 -*-
"""
Hello! This is a super smart Tic-Tac-Toe playing computer program! 🤖
It learns to play all by itself by playing thousands of games.
Imagine two robot friends playing Tic-Tac-Toe over and over to see who gets better.

Here are the cool tricks it uses:

1) Dueling DQN (The AI's Two-Part Brain):
   - One part of its brain asks: "Is this a good situation for me?" (Value Stream) 🤔
   - The other part asks: "Which move is the best one to make right now?" (Advantage Stream) 👉
   - By thinking about both, it makes really smart choices!

2) Double DQN (Learning without getting too excited):
   - Sometimes, an AI can get too optimistic about a move.
   - This trick helps it stay calm and realistic by using a "buddy" network to double-check its ideas. It's like asking a friend, "Is this really a good idea?" 🧐

3) NoisyNets (The Power of Being Random!):
   - Instead of always making the "best" move, this adds a little bit of silliness or "noise" to the AI's brain.
   - This helps the AI try new, unexpected moves, which is a great way to discover even better strategies! It's like trying a weird move just to see what happens. 🤪

4) Prioritized Experience Replay (PER - Remembering the Important Stuff):
   - The AI has a memory of its past games.
   - It pays extra attention to the games where something surprising happened (like an unexpected loss or win).
   - This helps it learn from its biggest mistakes and successes faster! It's like studying the hardest questions before a test. 📚

5) Self-Play (Learning from a Friend):
   - We have two AIs, one for 'X' and one for 'O'. They learn by playing against each other.
   - They only get a reward at the end of the game: +1 for winning, -1 for losing. Let the best bot win! 🏆

6) Tracking and Visualization (Seeing the AI Learn!):
   - We make cool charts and graphs to watch the AI get smarter over time.
   - We can see how often it wins, how long the games are, and how its brain is changing. It's like watching a progress report! 📈

I've added lots of comments so you can see exactly what's happening. Let's go!
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import math

# --- The Game World 🌍 ---
class TicTacToe:
    """
    This is the rulebook and the game board for Tic-Tac-Toe!
    It knows how the board looks, who's turn it is, and if someone has won.
    - 'X' is player 1
    - 'O' is player -1
    - An empty spot is 0
    The board is like a numbered keypad:
        0 | 1 | 2
        3 | 4 | 5
        6 | 7 | 8
    """
    def __init__(self):
        # Let's start a new game! The board is empty and no one has won yet.
        self.board = np.zeros(9, dtype=np.float32)
        self.current_winner = None

    def available_moves(self):
        """
        Where can I play? This looks for all the empty spots (the zeros).
        """
        return [i for i, spot in enumerate(self.board) if spot == 0]

    def make_move(self, square, letter):
        """
        Let's put an 'X' or an 'O' on the board.
        'square' is the spot (0-8) and 'letter' is the player (1 or -1).
        If the move is good, we check if it made someone win!
        """
        # Is this spot empty?
        if self.board[square] == 0:
            # Yep! Let's put our mark here.
            self.board[square] = letter
            # Did I just win the game with that move?! Let's check!
            if self.winner(square, letter):
                self.current_winner = letter
            return True # The move was successful!
        return False # Oops, that spot was already taken.

    def winner(self, square, letter):
        """
        Did we get three in a row? This checks if the last move won the game.
        It looks at the row, the column, and the diagonals.
        """
        # Check the row: Are all three spots in this row mine?
        row_ind = square // 3
        if all(self.board[row_ind*3 + i] == letter for i in range(3)): 
            return True

        # Check the column: Are all three spots in this column mine?
        col_ind = square % 3
        if all(self.board[col_ind + i*3] == letter for i in range(3)): 
            return True

        # Check the diagonals: Did I get a diagonal win?
        # Diagonals only matter if the move was in a corner or the center (0, 2, 4, 6, 8)
        if square % 2 == 0:
            # Top-left to bottom-right diagonal
            if all(self.board[i] == letter for i in [0, 4, 8]): 
                return True
            # Top-right to bottom-left diagonal
            if all(self.board[i] == letter for i in [2, 4, 6]): 
                return True

        # Nope, no winner yet.
        return False

    def get_state(self):
        """
        "Hey, what does the board look like right now?"
        This gives the AI a picture of the current game.
        """
        return self.board.copy()

    def reset(self):
        """
        New game! Let's wipe the board clean and start over.
        """
        self.board = np.zeros(9, dtype=np.float32)
        self.current_winner = None
        return self.get_state()

# --- The "Silly" Layer for Trying New Things 🤪 ---
class NoisyLinear(nn.Module):
    """
    This is a special part of the AI's brain that is a little bit "noisy" or random.
    Imagine if instead of always doing the perfect thing, you sometimes did something silly
    just to see what would happen. That's what this does!
    This randomness helps the AI explore new ideas and learn faster.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # These are the "normal" brain connections (the smart part)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # These are the "silly" or "noisy" connections that add randomness
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # We need a place to store the random noise itself
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        # Let's set up the brain and get some initial randomness
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Let's get the brain ready. We set the "smart" part to some good starting values,
        and the "silly" part to a small amount of randomness.
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """
        Let's shake things up! This creates new random numbers to make the AI try
        different things in the next step.
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Mix the input and output randomness together
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """
        A special math trick to make the randomness work well. It's like making sure
        our "silly" ideas are still helpful for learning.
        """
        return torch.randn(size).sign() * torch.randn(size).abs().sqrt()

    def forward(self, x):
        """
        When the AI is learning, we use both the smart part and the silly part.
        When the AI is just playing (not learning), it only uses the smart part.
        """
        if self.training:
            # Let's be smart AND silly!
            return F.linear(
                x, 
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon
            )
        else:
            # Time to be serious. Just use the smarts.
            return F.linear(x, self.weight_mu, self.bias_mu)

# --- The AI's Two-Part Brain 🧠 ---
class DuelingDQN(nn.Module):
    """
    This is the AI's brain! It has two special parts that work together.
    1. The "Value" part asks: "Overall, is this a good board for me to be in?" (V)
    2. The "Advantage" part asks: "How much better is this specific move compared to other moves?" (A)
    By combining V and A, the AI gets a super smart idea of the best move (Q-value).
    Q = V + A
    """
    def __init__(self):
        super(DuelingDQN, self).__init__()
        # First, the brain looks at the board and finds important patterns.
        self.feature_layer = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU()
        )
        # This is the "Value" part. It gives one score for the whole board.
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128), # We use our "silly" layers here!
            nn.ReLU(),
            NoisyLinear(128, 1)
        )
        # This is the "Advantage" part. It gives a score for each of the 9 possible moves.
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 128), # And here too!
            nn.ReLU(),
            NoisyLinear(128, 9)
        )

    def forward(self, x):
        """
        How the brain thinks:
        1. Look at the board and find patterns.
        2. The "Value" part thinks about how good the situation is.
        3. The "Advantage" part thinks about how good each move is.
        4. Combine them to make a final decision!
        """
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # The magic formula: Value + (Advantage - Average Advantage)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

    def reset_noise(self):
        """
        Tells all the "silly" parts of the brain to get new random ideas.
        "Time to shake things up again!"
        """
        for name, module in self.named_children():
            if 'stream' in name:
                for layer in module:
                    if isinstance(layer, NoisyLinear):
                        layer.reset_noise()

# --- The AI's Memory Diary 📔 ---
class ReplayMemory:
    """
    This is where the AI stores its memories of past games.
    It's like a diary where it writes down:
    - "The board looked like this..." (state)
    - "...then I made this move..." (action)
    - "...and then the board looked like this..." (next_state)
    - "...and this is what happened in the end!" (reward)
    
    But it's a special diary! It remembers the most surprising or important memories
    more clearly, so it can learn from them more often.
    """
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity # How many memories can I store?
        self.alpha = alpha # How much do I care about surprising memories?
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0  # New memories are super important!

    def push(self, state, action, next_state, reward):
        """
        Let's add a new memory to the diary!
        New memories get the highest priority, so the AI will definitely look at them soon.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (state, action, next_state, reward)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity # Go back to the start if the diary is full

    def sample(self, batch_size, beta=0.4):
        """
        Time to study! Let's pick some memories from the diary to learn from.
        The AI is more likely to pick the memories with high "priority" (the surprising ones).
        """
        if len(self.memory) == 0:
            return [], [], []

        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]

        # Figure out the chance of picking each memory
        probs = prios ** self.alpha
        probs /= probs.sum()

        # Pick a handful of memories based on their chances
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        # A little math to make sure learning from "important" memories doesn't mess up our brain
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """
        After studying a memory, the AI decides how "surprising" it was.
        If it was very surprising (the AI was wrong), we give it a higher priority
        so we will study it more in the future!
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
        self.max_priority = max(self.max_priority, np.max(batch_priorities))

    def __len__(self):
        return len(self.memory)

# --- The AI Player! 🤖 ---
class DQNAgent:
    """
    This is the AI player itself! It has a name ('X' or 'O'), a brain, and a memory.
    It knows how to pick a move and how to learn from its games to get better.
    """
    def __init__(self, player_letter, batch_size=128, gamma=0.99, tau=0.005, lr=1e-5):
        """
        Let's build our AI player!
        player_letter: Am I 'X' (+1) or 'O' (-1)?
        gamma: How much do I care about future rewards? (A high number means I think ahead!)
        tau: How slowly do I update my master plan? (A small number is safer)
        lr: How fast do I learn? (The learning rate)
        """
        self.player_letter = player_letter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        # Every player gets two brains! A "policy" brain for acting, and a "target" brain for thinking.
        self.policy_net = DuelingDQN().to(self.device)
        self.target_net = DuelingDQN().to(self.device)

        # At the start, both brains think exactly the same way.
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # This is the tool we use to help our brain learn from mistakes.
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

        # And every player gets their own memory diary!
        self.memory = ReplayMemory(10000)

    def select_action(self, state, available_moves):
        """
        Okay, it's my turn to move! What should I do?
        1. I'll look at the board.
        2. My brain will give a score to every possible move.
        3. I'll pick the move with the highest score!
        (My brain's "silliness" helps me explore, so I don't need to be random here).
        """
        with torch.no_grad(): # Just thinking, not learning right now.
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)

            # I can't play in a spot that's already taken! Let's ignore those.
            mask = torch.full(q_values.shape, -float('inf'), device=self.device)
            mask[0, available_moves] = 0

            # Apply the mask so I only consider good moves.
            q_values += mask

            # And the best move is... this one!
            return q_values.max(1)[1].view(1, 1)

    def optimize_model(self, beta):
        """
        Time to learn! 🧠 After a game, I sit down and study my memories.
        1. I grab a handful of important memories from my diary.
        2. I think about what I SHOULD have done.
        3. I compare it to what I ACTUALLY did.
        4. I update my brain to be a little bit smarter next time.
        5. I tell my brain to get new "silly" ideas for the next game.
        """
        if len(self.memory) < self.batch_size:
            return None # Not enough memories to study yet.

        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        if not transitions:
            return None

        # Let's organize our memories for studying.
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        state_batch = torch.tensor(np.array(batch_state), device=self.device, dtype=torch.float32)
        action_batch = torch.cat([a for a in batch_action])
        reward_batch = torch.tensor(np.array(batch_reward), device=self.device, dtype=torch.float32)
        weights_batch = torch.tensor(weights, device=self.device, dtype=torch.float32)

        # Which of these memories were from the end of a game?
        non_final_mask = torch.tensor([s is not None for s in batch_next_state], device=self.device, dtype=torch.bool)
        non_final_next_states_list = [s for s in batch_next_state if s is not None]

        if not non_final_next_states_list:
            non_final_next_states = torch.empty(0, 9, device=self.device, dtype=torch.float32)
        else:
            non_final_next_states = torch.tensor(np.array(non_final_next_states_list), device=self.device, dtype=torch.float32)

        # What score did my brain give the move I actually made?
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Let's figure out what the score SHOULD have been. This is the "Double DQN" trick!
        # 1. My main brain (policy_net) picks the best next move.
        # 2. My "buddy" brain (target_net) gives that move a score. This keeps me from being too optimistic!
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if len(non_final_next_states_list) > 0:
            with torch.no_grad():
                policy_next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                target_q = self.target_net(non_final_next_states).gather(1, policy_next_actions).squeeze()
                next_state_values[non_final_mask] = target_q

        # The "correct" score is the reward I got, plus the score of the next best move.
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # How wrong was I? The difference between what I thought and what was correct.
        td_error = (expected_state_action_values.unsqueeze(1) - state_action_values).abs()
        # This "error" becomes the new priority for the memory. If I was very wrong, it's very important!
        new_priorities = td_error.squeeze().detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, new_priorities)

        # Now, let's calculate the "loss" (how much I messed up) and use it to teach my brain.
        criterion = nn.SmoothL1Loss(reduction='none')
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = (loss.squeeze() * weights_batch).mean()

        # Time for the brain to learn!
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 10) # Don't learn too fast!
        self.optimizer.step()

        # Okay, learning session over. Let's get some new silly ideas!
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        return loss.item()

    def update_target_net(self):
        """
        It's important that my "buddy" brain doesn't change too fast.
        So, I'll slowly update it to be a little more like my main brain.
        It's like slowly teaching a friend your new strategy.
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + policy_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

# --- The Training Arena 🏟️ ---
def train(num_episodes=25000):
    """
    Welcome to the Tic-Tac-Toe Grand Championship!
    Here, we make two AI agents play against each other for thousands of games.
    With every game, they get a little bit smarter.
    We'll keep track of who wins, who loses, and how smart they're getting,
    and then we'll make some cool graphs to see their progress!
    """
    game = TicTacToe()
    agent1 = DQNAgent(player_letter=1)    # This is Agent X
    agent2 = DQNAgent(player_letter=-1)   # This is Agent O
    agents = {1: agent1, -1: agent2}

    # --- Let's keep score! This is the enhanced version with more stats! ---
    history = {
        'loss': [],         # How much did the AIs learn each game?
        'wins_x': [],       # How many times did X win?
        'wins_o': [],       # How many times did O win?
        'draws': [],        # How many ties?
        'episode_lengths': [],  # How many moves were in each game?
        'avg_max_q_x': []       # How confident is Agent X about the first move?
    }
    win_counts = {1: 0, -1: 0, 'draw': 0}

    # Let's watch how Agent X's confidence grows on an empty board.
    test_state = torch.tensor(np.zeros(9), dtype=torch.float32, device=agent1.device)

    # This helps our memory diary work better over time.
    beta_start = 0.4
    beta_frames = num_episodes
    beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

    # Let the games begin!
    for i_episode in range(num_episodes):
        # Start a fresh game.
        state = game.reset()
        current_player = 1       # X always goes first.
        done = False
        episode_transitions = []
        episode_length = 0

        # Keep playing until the game is over.
        while not done:
            episode_length += 1
            agent = agents[current_player]
            available_moves = game.available_moves()

            # Is the board full? If so, it's a draw!
            if not available_moves:
                done = True
                win_counts['draw'] += 1
                break

            # The current player chooses a move.
            action = agent.select_action(state, available_moves)

            # Let's remember the state and the action that was taken.
            episode_transitions.append({'state': state, 'action': action, 'player': current_player})

            # Make the move on the board.
            game.make_move(action.item(), current_player)
            next_state = game.get_state()

            # Did that move win the game?
            if game.current_winner is not None:
                done = True
                win_counts[game.current_winner] += 1

            # It's the next player's turn.
            state = next_state
            if not done:
                current_player = -current_player

        # Let's see how many moves that game took.
        history['episode_lengths'].append(episode_length)

        # Game over! Time to give out rewards.
        # Winner gets +1, loser gets -1. If it's a draw, nobody gets anything.
        final_reward = {1: 0, -1: 0}
        if game.current_winner is not None:
            final_reward[game.current_winner] = 1
            final_reward[-game.current_winner] = -1

        # Now, let's save all the moves from that game into each player's memory diary.
        for i, transition in enumerate(episode_transitions):
            player, s, a = transition['player'], transition['state'], transition['action']
            ns = None if i == len(episode_transitions) - 1 else episode_transitions[i+1]['state']
            # Only the very last move of the game for that player gets a reward.
            r = final_reward[player] if done and i == len(episode_transitions) - 1 else 0
            agents[player].memory.push(s, a, ns, r)

        # Okay, game's over. Time for both AIs to study and learn!
        beta = beta_by_frame(i_episode)
        loss1 = agents[1].optimize_model(beta)
        loss2 = agents[-1].optimize_model(beta)

        # And slowly update their "buddy" brains.
        agents[1].update_target_net()
        agents[-1].update_target_net()

        # Let's record how much they learned.
        avg_loss = np.mean([l for l in [loss1, loss2] if l is not None])
        if not np.isnan(avg_loss):
            history['loss'].append(avg_loss)

        # Every 100 games, let's check the score and print an update.
        if (i_episode + 1) % 100 == 0:
            history['wins_x'].append(win_counts[1])
            history['wins_o'].append(win_counts[-1])
            history['draws'].append(win_counts['draw'])
            win_counts = {1: 0, -1: 0, 'draw': 0} # Reset scores for the next 100 games.

            # How confident is Agent X feeling?
            with torch.no_grad():
                q_vals = agents[1].policy_net(test_state.unsqueeze(0))
                history['avg_max_q_x'].append(q_vals.max().item())

            print(f"Episode {i_episode+1}/{num_episodes}, Avg Loss: {np.mean(history['loss'][-100:]):.4f}")

    # --- Time to Make the Graphs! This is the ENHANCED Dashboard 📊 ---
    # We'll create a cool 2x3 grid to see everything we learned.
    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('How Our AI Agents Learned to Play Tic-Tac-Toe!', fontsize=20)

    # 1) How much did the AIs learn over time? (Lower is better!)
    loss_ma = np.convolve(history['loss'], np.ones(500)/500, mode='valid')
    axs[0, 0].plot(history['loss'], alpha=0.3, label='Learning per Game')
    axs[0, 0].plot(loss_ma, color='orange', label='Average Learning Trend')
    axs[0, 0].set_title('1. Learning Progress', fontsize=14)
    axs[0, 0].set_xlabel('Game Number')
    axs[0, 0].set_ylabel('Amount of Learning')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2) How confident did Agent X get about the first move? (Higher is better!)
    axs[0, 1].plot(history['avg_max_q_x'], color='purple')
    axs[0, 1].set_title('2. Agent X\'s Confidence at the Start', fontsize=14)
    axs[0, 1].set_xlabel('Game Chunk (x100)')
    axs[0, 1].set_ylabel('Confidence Score (Q-Value)')
    axs[0, 1].grid(True)

    # 3) How long were the games? (Longer games might mean they are playing smarter!)
    ep_len_ma = np.convolve(history['episode_lengths'], np.ones(500)/500, mode='valid')
    axs[0, 2].plot(history['episode_lengths'], alpha=0.3)
    axs[0, 2].plot(ep_len_ma, color='green', label='Average Game Length')
    axs[0, 2].set_title('3. Average Game Length', fontsize=14)
    axs[0, 2].set_xlabel('Game Number')
    axs[0, 2].set_ylabel('Number of Moves')
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # 4) Who won more? X, O, or was it a draw?
    total_games_chunk = np.array(history['wins_x']) + np.array(history['wins_o']) + np.array(history['draws'])
    total_games_chunk[total_games_chunk == 0] = 1 # Avoid dividing by zero!
    win_rate_x = np.array(history['wins_x']) / total_games_chunk
    win_rate_o = np.array(history['wins_o']) / total_games_chunk
    draw_rate = np.array(history['draws']) / total_games_chunk
    axs[1, 0].stackplot(
        range(len(win_rate_x)), win_rate_x, win_rate_o, draw_rate,
        labels=['X Wins', 'O Wins', 'Draws'],
        colors=['#1f77b4', '#d62728', '#2ca02c']
    )
    axs[1, 0].set_title('4. Who is Winning?', fontsize=14)
    axs[1, 0].set_xlabel('Game Chunk (x100)')
    axs[1, 0].set_ylabel('Percentage of Games')
    axs[1, 0].legend(loc='upper right')
    axs[1, 0].grid(True)

    # 5) Let's see the win rates as lines.
    axs[1, 1].plot(win_rate_x, label='X Win Rate', color='#1f77b4', marker='.')
    axs[1, 1].plot(win_rate_o, label='O Win Rate', color='#d62728', marker='.')
    axs[1, 1].set_title('5. Win Rates Over Time', fontsize=14)
    axs[1, 1].set_xlabel('Game Chunk (x100)')
    axs[1, 1].set_ylabel('Rate')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 6) Total wins and draws over all games played.
    cum_wins_x = np.cumsum(history['wins_x'])
    cum_wins_o = np.cumsum(history['wins_o'])
    cum_draws = np.cumsum(history['draws'])
    axs[1, 2].stackplot(
        range(len(cum_wins_x)), cum_wins_x, cum_wins_o, cum_draws,
        labels=['Total X Wins', 'Total O Wins', 'Total Draws'],
        colors=['#1f77b4', '#d62728', '#2ca02c']
    )
    axs[1, 2].set_title('6. Total Scoreboard', fontsize=14)
    axs[1, 2].set_xlabel('Game Chunk (x100)')
    axs[1, 2].set_ylabel('Total Number of Games')
    axs[1, 2].legend(loc='upper left')
    axs[1, 2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    # Let's start the training!
    # If you want a quick test, you can try a smaller number like 5000 games.
    train()
