import raylibpy as rl
import configs
from rubik import Rubik
from utils import generate_random_movements
from Agent import QLearningAgent
import numpy as np

# Initialize the window
rl.init_window(configs.window_w, configs.window_h, "ML_AISSAM_PROJECT")

rubik_cube = Rubik()
rotation_queue = []  # Start with an empty queue
scramble_moves = []  # track the scramble movements

rl.set_target_fps(configs.fps)

# Set the camera type (perspective or orthographic)
configs.camera.type = rl.CAMERA_PERSPECTIVE

# Button rectangles
scramble_button = rl.Rectangle(10, 10, 100, 30)
solve_button = rl.Rectangle(120, 10, 100, 30)
train_button = rl.Rectangle(230, 10, 100, 30)
test_button = rl.Rectangle(340, 10, 100, 30)

# Function to check if a mouse click is within a rectangle
def is_mouse_in_rect(rect):
    mouse_pos = rl.get_mouse_position()
    return (
        mouse_pos.x >= rect.x
        and mouse_pos.x <= rect.x + rect.width
        and mouse_pos.y >= rect.y
        and mouse_pos.y <= rect.y + rect.height
    )

# RL setup
state_size = 72
action_size = len(configs.rubiks_moves)
agent = QLearningAgent(state_size, action_size)

possible_moves = list(configs.rubiks_moves.keys())

def generate_scramble_moves(n):
    """Generates a list of random moves and records them for solving."""
    global scramble_moves
    rotation_queue = []
    scramble_moves = []  # Reset the list of recorded moves
    for _ in range(n):
        action = np.random.choice(list(configs.rubiks_moves.keys()))
        rotation_queue.append(configs.rubiks_moves[action])
        scramble_moves.append(action)  # record action name
    return rotation_queue

def get_reverse_moves():
    """Gets the reverse sequence of recorded scramble moves."""
    reversed_moves = []
    for move_name in reversed(scramble_moves):
        if move_name[-1] == '\'':
            reversed_moves.append(configs.rubiks_moves[move_name[:-1]])
        else:
            reversed_moves.append(configs.rubiks_moves[move_name + '\''])
    return reversed_moves

def training_loop(num_episodes=1000):
    for episode in range(num_episodes):
        rubik_cube.generate_rubik(2)
        rotation_queue = generate_random_movements(20)
        while rotation_queue:
            rotation_queue, _ = rubik_cube.handle_rotation(rotation_queue)
        
        state = hash(tuple(rubik_cube.get_state()))
        episode_reward = 0
        max_steps = 100

        for step in range(max_steps):
            action_index = agent.get_action(state)
            move_name = possible_moves[action_index]
            rotation_queue.append(configs.rubiks_moves[move_name])
            while rotation_queue:
                rotation_queue, animation_step = rubik_cube.handle_rotation(rotation_queue)
                if animation_step is not None: #play sound at the end of each rotation
                   play_sound()

            next_state = hash(tuple(rubik_cube.get_state()))
            reward = 1 if rubik_cube.is_solved() else -0.01
            agent.update_q_table(state, action_index, reward, next_state)
            state = next_state
            episode_reward += reward

            if rubik_cube.is_solved():
                break

        agent.decay_exploration_rate()


def test_loop(num_episodes=5):
    print("Starting Test Loop")
    agent.reset_exploration_rate()
    for episode in range(num_episodes):
        print(f"Testing episode: {episode}")
        rubik_cube.generate_rubik(2)
        rotation_queue = generate_random_movements(20)
        while rotation_queue:
            rotation_queue, _ = rubik_cube.handle_rotation(rotation_queue)
        
        state = hash(tuple(rubik_cube.get_state()))
        max_steps = 100

        for step in range(max_steps):
            action_index = agent.get_action(state)
            move_name = possible_moves[action_index]
            rotation_queue.append(configs.rubiks_moves[move_name])
            while rotation_queue:
                rotation_queue, animation_step = rubik_cube.handle_rotation(rotation_queue)
                if animation_step is not None:  #play sound at the end of each rotation
                  play_sound()

            state = hash(tuple(rubik_cube.get_state()))

            if rubik_cube.is_solved():
              print("Solved in episode", episode, step)
              break

# Load the sound
sound = rl.load_sound("cube_click.mp3")

# Function to play the sound
def play_sound():
    rl.play_sound(sound)


# Main rendering loop
while not rl.window_should_close():
    rotation_queue, animation_step = rubik_cube.handle_rotation(rotation_queue) #get animation step
    if animation_step is not None: #play sound at the end of each rotation
        play_sound()
    rl.update_camera(configs.camera, rl.CameraMode.CAMERA_THIRD_PERSON)

    rl.begin_drawing()
    rl.clear_background(rl.RAYWHITE)

    # Draw buttons
    rl.draw_rectangle_rec(scramble_button, rl.LIGHTGRAY)
    rl.draw_text("Scramble", int(scramble_button.x + 10), int(scramble_button.y + 10), 20, rl.BLACK)

    rl.draw_rectangle_rec(solve_button, rl.LIGHTGRAY)
    rl.draw_text("Solve", int(solve_button.x + 10), int(solve_button.y + 10), 20, rl.BLACK)

    rl.draw_rectangle_rec(train_button, rl.LIGHTGRAY)
    rl.draw_text("Train", int(train_button.x + 10), int(train_button.y + 10), 20, rl.BLACK)

    rl.draw_rectangle_rec(test_button, rl.LIGHTGRAY)
    rl.draw_text("Test", int(test_button.x + 10), int(test_button.y + 10), 20, rl.BLACK)

    # Handle button clicks
    if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT):
        if is_mouse_in_rect(scramble_button):
            rotation_queue = generate_scramble_moves(20)

        if is_mouse_in_rect(solve_button):
            if not rubik_cube.is_solved():
                print("Solving the Rubik's Cube by reversing!")
                rotation_queue = get_reverse_moves()


        if is_mouse_in_rect(train_button):
            training_loop(100)
        
        if is_mouse_in_rect(test_button):
            test_loop(5)


    rl.begin_mode3d(configs.camera)
    #rl.draw_grid(20, 1.0)

    # draw each cube of the Rubik's cube
    for i, cube in enumerate(rubik_cube.cubes):
        for cube_part in cube:
            position = rl.Vector3(cube[0].center[0], cube[0].center[1], cube[0].center[2])
            rl.draw_model(
                cube_part.model,
                position,
                2,
                cube_part.face_color
            )
    rl.end_mode3d()
    rl.end_drawing()

rl.unload_sound(sound)
rl.close_window()