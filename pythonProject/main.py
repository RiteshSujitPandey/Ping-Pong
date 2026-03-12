import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# ------------------ SETUP ------------------
game_width, game_height = 1280, 720

# Load paddle and ball images
paddle_img = cv2.imread(r"C:\Users\Ritesh\Downloads\piong.jpg", cv2.IMREAD_UNCHANGED)
ball_img = cv2.imread(r"C:\Users\Ritesh\Downloads\ball.png", cv2.IMREAD_UNCHANGED)

# Check image loading
if paddle_img is None or ball_img is None:
    print("❌ Error: Image paths are incorrect!")
    exit()

# Resize ball and paddle
ball_img = cv2.resize(ball_img, (40, 40))
paddle_img = cv2.resize(paddle_img, (50, 200))
h1, w1, _ = paddle_img.shape

# Ensure alpha channel
def add_alpha(img):
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        img = cv2.merge((b, g, r, alpha))
    return img

paddle_img = add_alpha(paddle_img)
ball_img = add_alpha(ball_img)

# Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not detected!")
    exit()

cap.set(3, game_width)
cap.set(4, game_height)

detector = HandDetector(detectionCon=0.75, maxHands=2)

# ------------------ GAME STATE ------------------
ballPos = [game_width // 2, game_height // 2]
speedX, speedY = 15, 15
score = [0, 0]
gameOver = False

left_paddle_y = game_height // 2
right_paddle_y = game_height // 2
left_paddle_x = 50
right_paddle_x = game_width - w1 - 50
smoothing = 0.3

# ------------------ MAIN LOOP ------------------
while True:
    success, frame = cap.read()
    if not success:
        continue

    img = cv2.flip(frame, 1)
    hands, _ = detector.findHands(img, flipType=False)

    # Draw center line
    cv2.line(img, (game_width // 2, 0), (game_width // 2, game_height), (0, 0, 0), 3)

    if not gameOver:
        # Track hands
        for hand in hands:
            fingers = detector.fingersUp(hand)
            if sum(fingers) >= 4:  # open hand
                x, y, w, h = hand['bbox']
                y1 = np.clip(y - h1 // 2, 0, game_height - h1)
                if x < game_width // 2:  # left paddle
                    left_paddle_y = int(left_paddle_y * (1 - smoothing) + y1 * smoothing)
                else:  # right paddle
                    right_paddle_y = int(right_paddle_y * (1 - smoothing) + y1 * smoothing)

        # Draw paddles
        img = cvzone.overlayPNG(img, paddle_img, (left_paddle_x, left_paddle_y))
        img = cvzone.overlayPNG(img, paddle_img, (right_paddle_x, right_paddle_y))

        # Ball collision with paddles
        if left_paddle_x < ballPos[0] < left_paddle_x + w1 and \
           left_paddle_y < ballPos[1] < left_paddle_y + h1:
            speedX = abs(speedX) + 1  # always bounce right, increase speed
            ballPos[0] = left_paddle_x + w1 + 5
            score[0] += 1

        if right_paddle_x < ballPos[0] < right_paddle_x + w1 and \
           right_paddle_y < ballPos[1] < right_paddle_y + h1:
            speedX = -(abs(speedX) + 1)  # always bounce left, increase speed
            ballPos[0] = right_paddle_x - 40
            score[1] += 1

        # Ball physics (top/bottom bounce)
        if ballPos[1] >= game_height - 30 or ballPos[1] <= 30:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        # Out-of-bounds check
        if ballPos[0] < 0 or ballPos[0] > game_width:
            gameOver = True

        # Draw ball and score
        img = cvzone.overlayPNG(img, ball_img, ballPos)
        cv2.putText(img, str(score[0]), (300, 80), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 5)
        cv2.putText(img, str(score[1]), (900, 80), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 0), 5)

    else:
        # Game Over screen
        cv2.putText(img, "GAME OVER", (450, 320), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)
        cv2.putText(img, "Press R to Restart", (460, 400), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(img, "Press Q to Quit", (480, 460), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 3)

    # Show window
    cv2.imshow("Ping Pong Game", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and gameOver:
        # Reset game
        ballPos = [game_width // 2, game_height // 2]
        speedX, speedY = 8, 8
        score = [0, 0]
        gameOver = False

cap.release()
cv2.destroyAllWindows()
