import pygame
import numpy as np
from neural_network import NeuralNetwork

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Paddle and Ball settings
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
BALL_SIZE = 15

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong with Neural Network - Play Mode")

# Font for displaying score
font = pygame.font.Font(None, 36)

# Define paddle, ball, and movement speed
paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
paddle_speed = 27
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_speed_x, ball_speed_y = 5, 5

# Adjust these values as needed
PREDICTION_FRAMES = 10  # Number of frames to look ahead

# Initialize the neural network and load weights
network = NeuralNetwork(input_size=6, hidden_size=12, output_size=1, weights_file='pong_weights.npz')

def draw_paddle(paddle_y):
    pygame.draw.rect(screen, WHITE, (20, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))

def draw_ball(ball_x, ball_y):
    pygame.draw.ellipse(screen, WHITE, (ball_x, ball_y, BALL_SIZE, BALL_SIZE))

def draw_score(score, high_score):
    score_text = font.render(f"Score: {score}", True, WHITE)
    high_score_text = font.render(f"High Score: {high_score}", True, WHITE)
    screen.blit(score_text, (WIDTH - 150, 10))
    screen.blit(high_score_text, (WIDTH - 180, 50))

def predict_ball_position(ball_x, ball_y, ball_speed_x, ball_speed_y, frames):
    future_x = ball_x + ball_speed_x * frames
    future_y = ball_y + ball_speed_y * frames
    
    # Account for bounces
    if future_y < 0 or future_y > HEIGHT:
        future_y = HEIGHT - abs(future_y % HEIGHT)
    
    return future_x, future_y

def move_paddle(ball_x, ball_y, ball_speed_x, ball_speed_y, paddle_y):
    future_x, future_y = predict_ball_position(ball_x, ball_y, ball_speed_x, ball_speed_y, PREDICTION_FRAMES)
    inputs = np.array([[
        ball_x / WIDTH, 
        ball_y / HEIGHT, 
        future_x / WIDTH,
        future_y / HEIGHT,
        paddle_y / HEIGHT,
        1 - (ball_x / WIDTH)
    ]])
    
    output = network.feedforward(inputs)
    target_y = output[0][0] * HEIGHT
    distance = target_y - (paddle_y + PADDLE_HEIGHT / 2)
    movement = np.clip(distance, -paddle_speed, paddle_speed)
    
    return max(min(paddle_y + movement, HEIGHT - PADDLE_HEIGHT), 0)

def reset_ball():
    return WIDTH // 2, HEIGHT // 2, ball_speed_x, np.random.choice([-1, 1]) * ball_speed_y

# Main game loop
running = True
clock = pygame.time.Clock()
score = 0
high_score = 0

ball_x, ball_y, ball_speed_x, ball_speed_y = reset_ball()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Ball movement and collision logic
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    if ball_y <= 0 or ball_y >= HEIGHT - BALL_SIZE:
        ball_speed_y = -ball_speed_y

    if ball_x >= WIDTH - BALL_SIZE:
        ball_speed_x = -ball_speed_x
        score += 1

    if ball_x <= 30 and paddle_y < ball_y < paddle_y + PADDLE_HEIGHT:
        ball_speed_x = abs(ball_speed_x)

    if ball_x <= 0:
        high_score = max(score, high_score)
        score = 0
        ball_x, ball_y, ball_speed_x, ball_speed_y = reset_ball()

    paddle_y = move_paddle(ball_x, ball_y, ball_speed_x, ball_speed_y, paddle_y)

    # Render the game
    screen.fill(BLACK)
    draw_paddle(paddle_y)
    draw_ball(ball_x, ball_y)
    draw_score(score, high_score)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()