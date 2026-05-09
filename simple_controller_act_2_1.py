#simple controller with onboard camera

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time

#configuration constants
DEBOUNCE_TIME = 0.1 #100 milliseconds
MAX_ANGLE = 0.5
MAX_SPEED = 250
SPEED_INCR = 5
ANGLE_INCR = 0.05

#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

#Image processing example
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

#Display image on onboard display
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# main
def main():
    speed = 10
    angle = 0.0
    last_press = {}

    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    display_img = Display("display_image")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Process and display image 
        grey_image = greyscale_cv2(image)
        display_image(display_img, grey_image)

        #to reduce rebounds
        current_time = time.time()

        # Read keyboard
        key=keyboard.getKey()

        if key in last_press and (current_time - last_press[key] < DEBOUNCE_TIME):
            continue # Ignore rebound

        #pressed key accepted, update
        last_press[key] = current_time

        if key == keyboard.UP: #up
            if speed < MAX_SPEED:
                speed += SPEED_INCR
                print("up")
        elif key == keyboard.DOWN: #down
            if speed >= SPEED_INCR:
                speed -= SPEED_INCR
                print("down")
        elif key == keyboard.RIGHT: #right
            #change_steer_angle(+1)
            angle += ANGLE_INCR
            if angle > MAX_ANGLE:
                angle = MAX_ANGLE
            print("right")
        elif key == keyboard.LEFT: #left
            #change_steer_angle(-1)
            angle -= ANGLE_INCR
            if angle < -MAX_ANGLE:
                angle = -MAX_ANGLE
            print("left")
        elif key == ord('A'):
            #filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)
            
        #update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()