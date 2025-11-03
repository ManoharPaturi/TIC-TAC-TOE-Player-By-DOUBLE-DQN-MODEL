// main.c
// Unbeatable Tic-Tac-Toe in C with raylib
// ---------------------------------------------
// This is the C version of the classic Minimax Tic-Tac-Toe AI.
// It uses the raylib library for a simple, modern graphical interface.

#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h> // For INT_MIN and INT_MAX
#include <time.h>   // For seeding random numbers

// =============================================================================
// PART 1: GAME ENGINE & STATE
// =============================================================================

#define HUMAN_PLAYER 'X'
#define AI_PLAYER 'O'

// Represents the state of the game board
typedef struct {
    char board[9];          // 'X', 'O', or ' ' for empty
    char current_winner;    // 'X', 'O', or ' ' if no winner yet
} GameState;

// Represents the result of a minimax calculation
typedef struct {
    int score;
    int position;
} MinimaxResult;

// Function Prototypes
void InitGame(GameState *game);
bool MakeMove(GameState *game, int square, char letter);
bool CheckWinner(GameState *game, int square, char letter);
bool IsBoardFull(const GameState *game);
MinimaxResult Minimax(GameState *state, char player);
int GetAIMove(GameState *game);


// Initialize or reset the game to its starting state
void InitGame(GameState *game) {
    for (int i = 0; i < 9; i++) {
        game->board[i] = ' ';
    }
    game->current_winner = ' ';
}

// Check if the board is completely full (tie condition)
bool IsBoardFull(const GameState *game) {
    for (int i = 0; i < 9; i++) {
        if (game->board[i] == ' ') {
            return false;
        }
    }
    return true;
}

// Place a letter on the board and check if it's a winning move
bool MakeMove(GameState *game, int square, char letter) {
    if (square >= 0 && square < 9 && game->board[square] == ' ') {
        game->board[square] = letter;
        if (CheckWinner(game, square, letter)) {
            game->current_winner = letter;
        }
        return true;
    }
    return false;
}

// An efficient check to see if the last move resulted in a win
bool CheckWinner(GameState *game, int square, char letter) {
    // Check row
    int row_ind = square / 3;
    if (game->board[row_ind * 3] == letter && game->board[row_ind * 3 + 1] == letter && game->board[row_ind * 3 + 2] == letter) return true;

    // Check column
    int col_ind = square % 3;
    if (game->board[col_ind] == letter && game->board[col_ind + 3] == letter && game->board[col_ind + 6] == letter) return true;

    // Check diagonals (only if the move is on a diagonal)
    if (square % 2 == 0) {
        if (game->board[0] == letter && game->board[4] == letter && game->board[8] == letter) return true;
        if (game->board[2] == letter && game->board[4] == letter && game->board[6] == letter) return true;
    }

    return false;
}

// =============================================================================
// PART 2: THE UNBEATABLE MINIMAX AI
// =============================================================================

// The core Minimax algorithm
MinimaxResult Minimax(GameState *state, char player) {
    char max_player = AI_PLAYER;
    char other_player = (player == 'X') ? 'O' : 'X';

    // Base cases: check for a terminal state (win, lose, tie)
    if (state->current_winner == other_player) {
        int available_moves_count = 0;
        for(int i = 0; i < 9; i++) {
            if(state->board[i] == ' ') available_moves_count++;
        }
        int score = (available_moves_count + 1) * (other_player == max_player ? 1 : -1);
        return (MinimaxResult){score, -1};
    }
    if (IsBoardFull(state)) {
        return (MinimaxResult){0, -1};
    }

    // Initialize the best move result
    MinimaxResult best;
    if (player == max_player) {
        best = (MinimaxResult){INT_MIN, -1}; // Maximize AI's score
    } else {
        best = (MinimaxResult){INT_MAX, -1}; // Minimize Human's score
    }

    // Loop through all available moves
    for (int i = 0; i < 9; i++) {
        if (state->board[i] == ' ') {
            // 1. Make a move on a temporary state
            state->board[i] = player;
            if (CheckWinner(state, i, player)) {
                state->current_winner = player;
            }

            // 2. Recurse to simulate the game after the move
            MinimaxResult sim_score = Minimax(state, other_player);

            // 3. Undo the move to explore other possibilities
            state->board[i] = ' ';
            state->current_winner = ' '; // Critical: reset winner after undoing
            sim_score.position = i;

            // 4. Update the best move based on the simulation's score
            if (player == max_player) { // AI is the maximizing player
                if (sim_score.score > best.score) {
                    best = sim_score;
                }
            } else { // Human is the minimizing player
                if (sim_score.score < best.score) {
                    best = sim_score;
                }
            }
        }
    }
    return best;
}

// Wrapper function to get the AI's final move
int GetAIMove(GameState *game) {
    int available_moves_count = 0;
    for(int i = 0; i < 9; i++) {
        if(game->board[i] == ' ') available_moves_count++;
    }

    if (available_moves_count == 9) {
        // If it's the first move, choose a random corner
        int corners[] = {0, 2, 6, 8};
        return corners[rand() % 4];
    }
    // Otherwise, calculate the best move using Minimax
    return Minimax(game, AI_PLAYER).position;
}


// =============================================================================
// PART 3: THE PLAYABLE GUI (using raylib)
// =============================================================================

int main(void) {
    // --- UI CONFIG ---
    const int screenWidth = 450;
    const int screenHeight = 550;
    const int boardSize = 420;
    const int cellSize = boardSize / 3;
    const int boardPadding = (screenWidth - boardSize) / 2;

    Color bgColor = { 44, 62, 80, 255 };     // Dark Blue-Gray (#2c3e50)
    Color gridColor = { 52, 73, 94, 255 };   // Lighter Blue-Gray (#34495e)
    Color xColor = { 52, 152, 219, 255 };    // Blue (#3498db)
    Color oColor = { 231, 76, 60, 255 };     // Red (#e74c3c)
    Color textColor = { 236, 240, 241, 255 };// Almost White (#ecf0f1)
    Color btnColor = { 26, 188, 156, 255 };  // Turquoise (#1abc9c)
    Color btnHoverColor = { 22, 160, 133, 255 }; // Greener Sea (#16a085)

    // --- INITIALIZATION ---
    InitWindow(screenWidth, screenHeight, "Tic-Tac-Toe AI | C + raylib");
    SetTargetFPS(60);
    srand(time(NULL)); // Seed random number generator

    GameState game;
    InitGame(&game);

    bool gameOver = false;
    char currentPlayer = HUMAN_PLAYER;
    char statusMessage[50] = "Your turn (X)";

    // --- GAME LOOP ---
    while (!WindowShouldClose()) {
        // --- UPDATE (Game Logic) ---
        if (!gameOver && currentPlayer == HUMAN_PLAYER) {
            if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
                Vector2 mousePos = GetMousePosition();
                int col = (mousePos.x - boardPadding) / cellSize;
                int row = (mousePos.y - boardPadding) / cellSize;
                int cellIndex = row * 3 + col;

                if (col >= 0 && col < 3 && row >= 0 && row < 3) {
                     if (MakeMove(&game, cellIndex, HUMAN_PLAYER)) {
                        if (game.current_winner != ' ') {
                            sprintf(statusMessage, "You Win! Congrats!");
                            gameOver = true;
                        } else if (IsBoardFull(&game)) {
                            sprintf(statusMessage, "It's a Tie!");
                            gameOver = true;
                        } else {
                            currentPlayer = AI_PLAYER;
                            sprintf(statusMessage, "AI is thinking...");
                        }
                    }
                }
            }
        } else if (!gameOver && currentPlayer == AI_PLAYER) {
            // AI's turn
            int aiMove = GetAIMove(&game);
            if (MakeMove(&game, aiMove, AI_PLAYER)) {
                if (game.current_winner != ' ') {
                    sprintf(statusMessage, "AI Wins! Better luck next time.");
                    gameOver = true;
                } else if (IsBoardFull(&game)) {
                    sprintf(statusMessage, "It's a Tie!");
                    gameOver = true;
                } else {
                    currentPlayer = HUMAN_PLAYER;
                    sprintf(statusMessage, "Your turn (X)");
                }
            }
        }

        // Handle New Game button click
        Rectangle resetButton = { screenWidth / 2 - 75, 480, 150, 50 };
        bool buttonHover = CheckCollisionPointRec(GetMousePosition(), resetButton);
        if (buttonHover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
            InitGame(&game);
            gameOver = false;
            currentPlayer = HUMAN_PLAYER;
            sprintf(statusMessage, "Your turn (X)");
        }


        // --- DRAW (Render Graphics) ---
        BeginDrawing();
        ClearBackground(bgColor);

        // Draw grid lines
        for (int i = 1; i < 3; i++) {
            DrawLineEx((Vector2){boardPadding + i * cellSize, boardPadding}, (Vector2){boardPadding + i * cellSize, boardPadding + boardSize}, 8, gridColor);
            DrawLineEx((Vector2){boardPadding, boardPadding + i * cellSize}, (Vector2){boardPadding + boardSize, boardPadding + i * cellSize}, 8, gridColor);
        }

        // Draw X's and O's
        for (int i = 0; i < 9; i++) {
            int row = i / 3;
            int col = i % 3;
            float centerX = boardPadding + col * cellSize + cellSize / 2.0f;
            float centerY = boardPadding + row * cellSize + cellSize / 2.0f;
            
            if (game.board[i] == 'X') {
                DrawText("X", centerX - 30, centerY - 60, 120, xColor);
            } else if (game.board[i] == 'O') {
                DrawText("O", centerX - 35, centerY - 60, 120, oColor);
            }
        }

        // Draw status message
        DrawText(statusMessage, screenWidth / 2 - MeasureText(statusMessage, 20) / 2, 20, 20, textColor);

        // Draw New Game button
        DrawRectangleRec(resetButton, buttonHover ? btnHoverColor : btnColor);
        DrawText("New Game", resetButton.x + resetButton.width / 2 - MeasureText("New Game", 20) / 2, resetButton.y + 15, 20, textColor);

        EndDrawing();
    }

    // --- CLEANUP ---
    CloseWindow();
    return 0;
}