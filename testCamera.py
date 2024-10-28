import cv2
import tkinter as tk
import time
from PIL import Image, ImageTk

object_start_time = None
capture_duration = 5
count = 0

def start_video():
    cap = cv2.VideoCapture(0)
    
    def update_frame():
        global object_start_time, count
        
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            height, width, _ = frame.shape
            
            square_size = 500
            top_left_x = (width - square_size) // 2
            top_left_y = (height - square_size) // 2
            bottom_right_x = top_left_x + square_size
            bottom_right_y = top_left_y + square_size
            
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
            _, threshold = cv2.threshold(blurred_frame, 60, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            object_inside_square = False
            
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                if (top_left_x < x < bottom_right_x and top_left_y < y < bottom_right_y):
                    object_inside_square = True
                    break
            
            if object_inside_square:
                if object_start_time is None:
                    object_start_time = time.time()
                else:
                    elapsed_time = time.time() - object_start_time
                    if elapsed_time >= capture_duration:
                        captured_image = Image.fromarray(frame)
                        captured_image.save(f"captured_image_{count}.png")
                        print(f"Image captured and saved as captured_image_{count}.png")
                        count += 1
                        object_start_time = None
            else:
                object_start_time = None
            
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        
        video_label.after(10, update_frame)
    
    update_frame()

root = tk.Tk()
root.title("Image Capture Testing")

video_label = tk.Label(root)
video_label.pack()

start_video()

root.mainloop()
