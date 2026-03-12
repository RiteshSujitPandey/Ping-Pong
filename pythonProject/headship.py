import cv2
import mediapipe as mp
import pygame
import random
import sys

# ---------- Game Settings ----------
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 800
SPACESHIP_SPEED = 10
ASTEROID_SPEED = 7
ASTEROID_FREQ = 25  # frames

# ---------- Initialize Pygame ----------
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AstroDodger Head Control")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 30)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# ---------- Load Assets ----------
SPACESHIP_IMG = pygame.Surface((50, 50))
SPACESHIP_IMG.fill((0, 255, 0))
ASTEROID_IMG = pygame.Surface((50, 50))
ASTEROID_IMG.fill((128, 128, 128))

# ---------- OpenCV & MediaPipe Setup ----------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# ---------- Game Variables ----------
spaceship_x = SCREEN_WIDTH // 2 - 25
spaceship_y = SCREEN_HEIGHT - 100
asteroids = []
frame_count = 0
score = 0

# ---------- Game Loop ----------
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found.")
        break

    # Flip camera for natural movement
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Update spaceship x based on nose position
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]  # Nose tip
            nose_x = int(nose.x * SCREEN_WIDTH)
            spaceship_x = max(0, min(SCREEN_WIDTH - 50, nose_x - 25))

    # ---------- Event Handling ----------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ---------- Asteroid Logic ----------
    frame_count += 1
    if frame_count % ASTEROID_FREQ == 0:
        asteroid_x = random.randint(0, SCREEN_WIDTH - 50)
        asteroids.append([asteroid_x, -50])

    for asteroid in asteroids:
        asteroid[1] += ASTEROID_SPEED

    # Collision Detection
    for asteroid in asteroids:
        if (spaceship_x < asteroid[0] + 50 and spaceship_x + 50 > asteroid[0] and
            spaceship_y < asteroid[1] + 50 and spaceship_y + 50 > asteroid[1]):
            running = False

    # Remove off-screen asteroids and update score
    asteroids = [a for a in asteroids if a[1] < SCREEN_HEIGHT]
    score += 1

    # ---------- Drawing ----------
    screen.fill(BLACK)
    screen.blit(SPACESHIP_IMG, (spaceship_x, spaceship_y))
    for asteroid in asteroids:
        screen.blit(ASTEROID_IMG, asteroid)

    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))
    pygame.display.flip()
    clock.tick(30)

# ---------- Game Over ----------
cap.release()
pygame.quit()
print(f"Game Over! Your score: {score}")
sys.exit()
