# Pong AI

## Overview
Pong AI is a simple implementation of the classic Pong game using Python and Pygame, enhanced with a neural network that allows the paddle to learn and improve its performance over time. The project demonstrates the integration of machine learning with game development, providing a fun and educational experience.

## Features
- Classic Pong gameplay with a paddle and a ball.
- Neural network that predicts the ball's future position and adjusts the paddle's movement accordingly.
- Training mechanism that allows the AI to learn from its gameplay.
- Score tracking and high score management.
- Ability to save and load neural network weights.

## Requirements
- Python 3.x
- Pygame
- NumPy
- A neural network implementation (provided in `neural_network.py`)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mattiemateo/PongAI.git
   cd PongAI
   ```

2. Install the required packages:
   ```bash
   pip install pygame numpy
   ```

3. Ensure you have the `neural_network.py` file in the project directory.

## Usage
1. Run the game:
   ```bash
   python train_pong.py
   ```

2. Control the paddle using the following keys:
   - **Up Arrow**: Move paddle up
   - **Down Arrow**: Move paddle down
   - **S**: Save the neural network weights
   - **L**: Load the neural network weights

3. The game will automatically train the neural network after a specified number of games played.

## Game Mechanics
- The paddle follows the ball's predicted future position based on its current speed and direction.
- The AI uses a neural network to determine the optimal paddle position, learning from past games.
- The game keeps track of the score and high score, saving the weights of the neural network when certain conditions are met.

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the creators of Pygame for providing a great library for game development.
- Special thanks to the machine learning community for the resources and knowledge shared.

## Contact
For any inquiries or feedback, please reach out to [your-email@example.com].
