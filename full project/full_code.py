import pygame
import sys
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from ultralytics import YOLO

# Initialize Pygame
pygame.init()

# Set up the display
screen_width, screen_height = 1200, 700
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Escape Room")

# Function to load and resize an image
def load_and_resize(image_path):
    image = pygame.image.load(image_path)
    return pygame.transform.scale(image, (screen_width, screen_height))

# Load assets
loading_gif = load_and_resize('ER/gii.gif')
start_screen = load_and_resize('ER/kkk.png')
new1_img = load_and_resize('ER/kkk.png')
new2_img = load_and_resize('ER/9.jpg')
new3_img = load_and_resize('ER/10.jpg')
new4_img = load_and_resize('ER/11.jpg')
new5_img = load_and_resize('ER/new5_img.png')  


def show_loading_screen():
    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < 5000:  # 20 seconds
        screen.blit(loading_gif, (0, 0))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

def show_start_screen():
    screen.blit(start_screen, (0, 0))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                waiting = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

def play_video(file_path, screen):
    clip = VideoFileClip(file_path)
    clip_duration = clip.duration

    start_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - start_time < clip_duration * 1000:
        elapsed = (pygame.time.get_ticks() - start_time) / 1000
        frame = clip.get_frame(elapsed)
        frame = np.flipud(np.rot90(frame))  # Correct orientation
        frame = pygame.surfarray.make_surface(frame)
        frame = pygame.transform.scale(frame, (screen_width, screen_height))
        screen.blit(frame, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                clip.close()
                pygame.quit()
                sys.exit()

    clip.close()

def show_image_and_wait(image):
    screen.blit(image, (0, 0))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                waiting = False
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

def open_camera_and_wait(room1_index):
    cap = cv2.VideoCapture(0)  # Open the default camera
    font = cv2.FONT_HERSHEY_SIMPLEX
    message_displayed = False

    
    # room1_index = 0
    
    model = YOLO('yolov8m.pt')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, stream=True)
            cls_name = None
            for r in results:
                boxes = r.boxes
                for bbox in boxes:
                    x1, y1, x2, y2 = bbox.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls_idx = int(bbox.cls[0])
                    cls_name = model.names[cls_idx]
                    conf = round(float(bbox.conf[0]), 2)
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (225, 0, 0), 4)
                    # cv2.putText(frame, f'{cls_name} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                

            # if message_displayed:
            #     cv2.putText(frame, 'TRY again', (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)  # Rotate the frame first
            frame = pygame.surfarray.make_surface(frame)
            screen.blit(frame, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        message_displayed = True
                    elif event.key == pygame.K_RETURN:
                        return
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()        

            if room1_index == 0:
                if cls_name == "bottle" and conf > 0.5:
                    room1_index = room1_index + 1
                    return

            if room1_index ==1:
                if cls_name == "book" and conf > 0.5:
                    room1_index = 2
                    return
            

            if room1_index ==2:
                if cls_name == "cell phone": #and conf > 0.5:
                   # room1_index = room1_index+1
                    return
                
   
    finally:
        cap.release()
        
def show_end_image_and_check_input(image):
    screen.blit(image, (0, 0))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    return True  # Restart the game
                elif event.key == pygame.K_q:
                    return False  # Quit the game
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

def main():
    i=0
    while True:  # Game loop
        show_loading_screen()
        show_start_screen()
        play_video('ER/mmm.mp4', screen)
        show_image_and_wait(new1_img)

        for image in [new2_img, new3_img, new4_img]:
            show_image_and_wait(image)
            
            open_camera_and_wait(i)
            i=i+1

        play_video('ER/vvv.mp4', screen)

        # Show end game image and wait for input
        restart_game = show_end_image_and_check_input(new5_img)
        if not restart_game:
            print("Game Over")
            break

main()