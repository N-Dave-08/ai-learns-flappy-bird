import pygame
import sys
import random
import neat
import pickle
import os

# --- 1. CONFIGURATION (Must match training physics) ---
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.25
BIRD_JUMP = -6
PIPE_SPEED = 4
PIPE_GAP = 150

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Flappy Bird - AI Showcase")
font = pygame.font.SysFont("Arial", 30)

# --- 2. ASSET LOADING ---
def load_image(path, w, h):
    return pygame.transform.scale(pygame.image.load(path).convert_alpha(), (w, h))

BIRD_IMGS = [load_image(f'assets/images/flappy0{i}.png', 50, 35) for i in range(6)]
PIPE_IMG = load_image('assets/images/pipe.png', 70, 500)
BASE_IMG = load_image('assets/images/ground.png', SCREEN_WIDTH, 100)
BG_IMG = load_image('assets/images/mounts.png', SCREEN_WIDTH, 300)

# --- 3. GAME CLASSES ---
class Bird:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vel = 0
        self.img_count = 0
        self.img = BIRD_IMGS[0]
        self.rect = self.img.get_rect(center=(x, y))

    def jump(self):
        self.vel = BIRD_JUMP

    def move(self):
        self.vel += GRAVITY
        self.y += self.vel
        self.rect.centery = self.y

    def draw(self, win):
        self.img_count = (self.img_count + 0.2) % 6
        self.img = BIRD_IMGS[int(self.img_count)]
        win.blit(self.img, self.rect)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(150, 400)
        self.top = self.height - 500
        self.bottom = self.height + PIPE_GAP
        self.pipe_top = pygame.transform.flip(PIPE_IMG, False, True)
        self.pipe_bottom = PIPE_IMG
        self.passed = False

    def move(self):
        self.x -= PIPE_SPEED

    def draw(self, win):
        win.blit(self.pipe_top, (self.x, self.top))
        win.blit(self.pipe_bottom, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = pygame.mask.from_surface(bird.img)
        top_mask = pygame.mask.from_surface(self.pipe_top)
        bottom_mask = pygame.mask.from_surface(self.pipe_bottom)
        t_offset = (self.x - bird.rect.left, self.top - bird.rect.top)
        b_offset = (self.x - bird.rect.left, self.bottom - bird.rect.top)
        return bird_mask.overlap(top_mask, t_offset) or bird_mask.overlap(bottom_mask, b_offset)

# --- 4. THE PLAY FUNCTION ---
def play_ai(config_path):
    # Load the saved "brain"
    try:
        with open("best_bird.pkl", "rb") as f:
            winner_genome = pickle.load(f)
    except FileNotFoundError:
        print("Error: best_bird.pkl not found. Run main.py first to train a bird!")
        return

    # Set up the Neural Network
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    # Initialize Game Objects
    bird = Bird(100, 300)
    pipes = [Pipe(500)]
    score = 0
    run = True

    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()

        # Logic: Which pipe is the AI looking at?
        pipe_ind = 0
        if len(pipes) > 1 and bird.x > pipes[0].x + 70:
            pipe_ind = 1

        # The AI makes a decision
        # Inputs: 1. Bird Y, 2. Distance to Top, 3. Distance to Bottom
        output = net.activate((bird.y, 
                               abs(bird.y - pipes[pipe_ind].height), 
                               abs(bird.y - pipes[pipe_ind].bottom)))

        if output[0] > 0.5:
            bird.jump()

        # Update Positions
        bird.move()
        add_pipe = False
        rem = []
        for pipe in pipes:
            pipe.move()
            if pipe.collide(bird):
                print(f"Final Score: {score}")
                run = False # Game Over

            if not pipe.passed and bird.x > pipe.x:
                pipe.passed = True
                add_pipe = True

            if pipe.x + 70 < 0:
                rem.append(pipe)

        if add_pipe:
            score += 1
            pipes.append(Pipe(500))

        for r in rem:
            pipes.remove(r)

        # Check Boundaries
        if bird.y + 35 >= 500 or bird.y < 0:
            print(f"Final Score: {score}")
            run = False

        # --- DRAWING ---
        screen.fill((112, 197, 206))
        screen.blit(BG_IMG, (0, 200))
        for pipe in pipes:
            pipe.draw(screen)
        screen.blit(BASE_IMG, (0, 500))
        bird.draw(screen)

        # Draw Score UI
        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 50))
        
        pygame.display.update()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    play_ai(config_path)