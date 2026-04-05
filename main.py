import pygame
import sys
import random
import neat
import os
import pickle

# --- 1. CONFIGURATION ---
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.25
BIRD_JUMP = -6
PIPE_SPEED = 4
PIPE_GAP = 150
GEN = 0 # Track which generation we are on

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# --- 2. ASSET LOADING ---
def load_image(path, w, h):
    return pygame.transform.scale(pygame.image.load(path).convert_alpha(), (w, h))

BIRD_IMGS = [load_image(f'assets/images/flappy0{i}.png', 50, 35) for i in range(6)]
PIPE_IMG = load_image('assets/images/pipe.png', 70, 500)
BASE_IMG = load_image('assets/images/ground.png', SCREEN_WIDTH, 100)
BG_IMG = load_image('assets/images/mounts.png', SCREEN_WIDTH, 300)

# --- 3. CLASSES ---
class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
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

        top_offset = (self.x - bird.rect.left, self.top - bird.rect.top)
        bottom_offset = (self.x - bird.rect.left, self.bottom - bird.rect.top)

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True

        return False

# --- 4. MAIN NEAT EVALUATION ---
def eval_genomes(genomes, config):
    global GEN
    GEN += 1
    nets = []
    ge = []
    birds = []

    # Initialize genomes
    for genome_id, genome in genomes:
        genome.fitness = 0 
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(100, 300))
        ge.append(genome)

    pipes = [Pipe(500)]
    score = 0
    run = True

    while run and len(birds) > 0:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Determine which pipe to look at
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + 70:
                pipe_ind = 1

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1 # Reward for staying alive
            bird.move()

            # Feed inputs to Neural Network
            # 1. Bird Y, 2. Dist to Top Pipe, 3. Dist to Bottom Pipe
            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:
                bird.jump()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1 # Penalty for dying
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and bird.x > pipe.x:
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + 70 < 0:
                rem.append(pipe)

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5 # Big reward for passing a pipe
            pipes.append(Pipe(500))

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):
            if bird.y + 35 >= 500 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        # --- DRAWING ---
        screen.fill((112, 197, 206))
        screen.blit(BG_IMG, (0, 200))
        for pipe in pipes:
            pipe.draw(screen)
        screen.blit(BASE_IMG, (0, 500))
        for bird in birds:
            bird.draw(screen)
        
        pygame.display.update()

# --- 5. RUN NEAT ---
def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # This runs the training and returns the smartest bird ever found
    winner = p.run(eval_genomes, 50) # Run for 50 generations

    # --- NEW: SAVE THE BEST BIRD ---
    with open("best_bird.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("\nTraining Complete! Best bird saved to best_bird.pkl")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)