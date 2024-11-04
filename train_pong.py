import pygame
import time
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
pygame.display.set_caption("Pong with Neural Network")

# Define paddle, ball, and movement speed
paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2
paddle_speed = 27  # Increased from 15
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_speed_x, ball_speed_y = 5, 5  # Constant speed

# Adjust these values as needed
PREDICTION_FRAMES = 10  # Number of frames to look ahead

# Initialize the neural network
network = NeuralNetwork(input_size=6, hidden_size=12, output_size=1, weights_file='pong_weights.npz')

# Variables for collecting training data
training_inputs = []
training_outputs = []
games_played = 0
max_games_before_training = 3

# Score variables
score = 0
high_score = 0

# Font for displaying score
font = pygame.font.Font(None, 36)

# Add these global variables at the top of your file, after other initializations
last_target_y = HEIGHT / 2
smoothing_factor = 0.2  # Adjust this value between 0 and 1 to change smoothing strength

def draw_paddle(paddle_y):
    pygame.draw.rect(screen, WHITE, (20, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))

def draw_ball(ball_x, ball_y):
    pygame.draw.ellipse(screen, WHITE, (ball_x, ball_y, BALL_SIZE, BALL_SIZE))

def draw_score():
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
    global last_target_y  # We'll use this to smooth the paddle movement
    
    current_distance = ball_x - PADDLE_WIDTH
    future_x, future_y = predict_ball_position(ball_x, ball_y, ball_speed_x, ball_speed_y, PREDICTION_FRAMES)
    urgency = max(0, 1 - (current_distance / WIDTH))
    
    inputs = np.array([[
        ball_x / WIDTH, 
        ball_y / HEIGHT, 
        future_x / WIDTH,
        future_y / HEIGHT,
        paddle_y / HEIGHT,
        urgency
    ]])
    
    output = network.feedforward(inputs)
    
    training_inputs.append(inputs[0])
    ideal_y = future_y - PADDLE_HEIGHT / 2
    training_outputs.append([ideal_y / HEIGHT])
    
    target_y = output[0][0] * HEIGHT
    
    # Smooth the target_y using the last_target_y
    smoothed_target_y = last_target_y + smoothing_factor * (target_y - last_target_y)
    last_target_y = smoothed_target_y
    
    distance = smoothed_target_y - (paddle_y + PADDLE_HEIGHT / 2)
    
    # Use urgency to determine paddle speed, but cap the maximum speed
    max_speed = paddle_speed * 2  # Adjust this value to change the maximum speed
    movement = np.clip(distance, -max_speed, max_speed) * (1 + urgency)
    
    paddle_y = max(min(paddle_y + movement, HEIGHT - PADDLE_HEIGHT), 0)
    
    return paddle_y

def reset_ball():
    return WIDTH // 2, HEIGHT // 2, ball_speed_x, np.random.choice([-1, 1]) * ball_speed_y

# Main game loop
running = True
clock = pygame.time.Clock()

ball_x, ball_y, ball_speed_x, ball_speed_y = reset_ball()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # Detect when "s" is pressed to save the weights
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                network.save_weights()
                print("Weights saved successfully.")
                time.sleep(2)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                network.load_weights()
                print("Weights loaded successfully.")
                time.sleep(2)

    # Ball movement and collision logic
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    if ball_y <= 0 or ball_y >= HEIGHT - BALL_SIZE:
        ball_speed_y = -ball_speed_y

    if ball_x >= WIDTH - BALL_SIZE:
        ball_speed_x = -ball_speed_x
        score += 1

        # Save weights if the score exceeds 1000
        if score > 1000:
            network.save_weights()

    if ball_x <= 30 and paddle_y < ball_y < paddle_y + PADDLE_HEIGHT:
        ball_speed_x = abs(ball_speed_x)
        urgency = max(0, 1 - ((30 - ball_x) / 30))
        training_outputs[-1] = [1.0 + urgency]

    if ball_x <= 0:
        high_score = max(score, high_score)
        score = 0
        ball_x, ball_y, ball_speed_x, ball_speed_y = reset_ball()
        games_played += 1
        
        if games_played % max_games_before_training == 0:
            print("Training network...")
            training_inputs = np.array(training_inputs)
            training_outputs = np.array(training_outputs)
            
            noise = np.random.normal(0, 0.05, training_inputs.shape)
            training_inputs += noise
            
            network.train(training_inputs, training_outputs, learning_rate=0.01, epochs=1000)
            training_inputs = []
            training_outputs = []
            print(f"Trained after {games_played} games. High score: {high_score}")

    paddle_y = move_paddle(ball_x, ball_y, ball_speed_x, ball_speed_y, paddle_y)
    print(f"Ball: ({ball_x}, {ball_y}), Paddle: {paddle_y}")  # Debug print

    screen.fill(BLACK)
    draw_paddle(paddle_y)
    draw_ball(ball_x, ball_y)
    draw_score()
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
