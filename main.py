import pygame
import sys
import random

# --- 1. CONFIGURATION (Your "State" Constants) ---
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.25
BIRD_JUMP = -6
PIPE_SPEED = 3
PIPE_GAP = 170  # Distance between top and bottom pipes
SPAWN_RATE = 1400 # Milliseconds between pipes

# --- 2. INITIALIZATION ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Flappy Bird - Python Edition")

# --- 3. ASSET LOADING & SCALING ---
def load_image(path, scale_width=None, scale_height=None):
    img = pygame.image.load(path).convert_alpha()
    if scale_width and scale_height:
        img = pygame.transform.scale(img, (scale_width, scale_height))
    return img

# Bird Frames (Cycling flappy00 to flappy05)
bird_frames = [load_image(f'assets/images/flappy0{i}.png', 50, 35) for i in range(6)]
bird_index = 0
bird_surface = bird_frames[bird_index]
bird_rect = bird_surface.get_rect(center=(100, SCREEN_HEIGHT // 2))

# Obstacles & Background
pipe_surface = load_image('assets/images/pipe.png', 70, 500)
mount_surface = load_image('assets/images/mounts.png', SCREEN_WIDTH, 300)
ground_surface = load_image('assets/images/ground.png', SCREEN_WIDTH, 100)

# --- 4. GAME VARIABLES ---
bird_movement = 0
pipe_list = []
score = 0
game_active = True

# Custom Timer for Spawning Pipes
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWNPIPE, SPAWN_RATE)

# --- 5. HELPER FUNCTIONS ---
def create_pipe():
    # Randomly pick the vertical center of the gap
    random_pipe_pos = random.randint(200, 400)
    bottom_pipe = pipe_surface.get_rect(midtop=(SCREEN_WIDTH + 50, random_pipe_pos + (PIPE_GAP / 2)))
    top_pipe = pipe_surface.get_rect(midbottom=(SCREEN_WIDTH + 50, random_pipe_pos - (PIPE_GAP / 2)))
    return bottom_pipe, top_pipe

def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= PIPE_SPEED
    # Clean up off-screen pipes (Garbage Collection)
    return [pipe for pipe in pipes if pipe.right > -50]

def draw_pipes(pipes):
    for pipe in pipes:
        if pipe.bottom >= SCREEN_HEIGHT: # It's a bottom pipe
            screen.blit(pipe_surface, pipe)
        else: # It's a top pipe - Flip it!
            flip_pipe = pygame.transform.flip(pipe_surface, False, True)
            screen.blit(flip_pipe, pipe)

def check_collision(pipes):
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            return False
    if bird_rect.top <= -100 or bird_rect.bottom >= SCREEN_HEIGHT - 100:
        return False
    return True

# --- 6. MAIN GAME LOOP ---
while True:
    # --- EVENT HANDLING ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if game_active:
                    bird_movement = BIRD_JUMP
                else:
                    # Reset Game
                    game_active = True
                    pipe_list.clear()
                    bird_rect.center = (100, SCREEN_HEIGHT // 2)
                    bird_movement = 0
                    score = 0

        if event.type == SPAWNPIPE and game_active:
            pipe_list.extend(create_pipe())

    # --- DRAWING & LOGIC ---
    # 1. Background (Static Sky Color + Mountains)
    screen.fill((112, 197, 206))
    screen.blit(mount_surface, (0, SCREEN_HEIGHT - 400))

    if game_active:
        # 2. Bird Physics & Animation
        bird_movement += GRAVITY
        bird_rect.centery += bird_movement
        
        # Animate bird wing flap
        bird_index = (bird_index + 0.2) % 6 # Using float increment for smoother flap
        bird_surface = bird_frames[int(bird_index)]

        screen.blit(bird_surface, bird_rect)

        
        # 3. Pipes
        pipe_list = move_pipes(pipe_list)
        draw_pipes(pipe_list)
        
        # 4. Check Collisions
        game_active = check_collision(pipe_list)
    else:
        # Game Over Screen Logic (You can add text here)
        pass

    # 5. Ground (Draw last so it's on top of pipes)
    screen.blit(ground_surface, (0, SCREEN_HEIGHT - 100))

    # --- REFRESH ---
    pygame.display.update()
    clock.tick(60) # Locked 60 FPS