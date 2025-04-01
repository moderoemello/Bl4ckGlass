import json
import os
import subprocess
import time
import pyautogui
import pytesseract
from PIL import Image
import cv2
import numpy as np

# === Load Config ===
with open("config.json") as f:
    config = json.load(f)

chrome_profile = config["chrome_profile"]
resume_path = config["resume_path"]
job_keywords = config["job_keywords"]
job_location = config["job_location"]
email = config["login"]["email"]
password = config["login"]["password"]

# === Launch Chrome with Profile ===
def launch_chrome():
    subprocess.Popen([
        "google-chrome",
        f'--profile-directory={chrome_profile}',
        "--new-window",
        "https://www.linkedin.com"
    ])
    time.sleep(5)

# === Capture Screenshot ===
def capture_screen():
    screenshot = pyautogui.screenshot()
    path = "screen.png"
    screenshot.save(path)
    return path

# === Vision Agent: Find Image on Screen ===
def find_element(template_path, screenshot_path="screen.png", threshold=0.75):
    screen = cv2.imread(screenshot_path)
    template = cv2.imread(template_path)
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        x, y = max_loc
        h, w = template.shape[:2]
        return x + w // 2, y + h // 2
    return None

# === Input Agent: Move & Click ===
def click(x, y):
    pyautogui.moveTo(x, y, duration=0.5)
    pyautogui.click()

# === Main Workflow ===
def main():
    launch_chrome()
    print("[+] Chrome launched")

    time.sleep(10)  # wait for LinkedIn to load

    screen_path = capture_screen()
    print("[+] Screen captured")

    # Example: look for Easy Apply button
    button_path = "assets/easy_apply.png"
    coords = find_element(button_path, screen_path)
    
    if coords:
        print(f"[+] Found Easy Apply at {coords}")
        click(*coords)
    else:
        print("[-] Easy Apply not found")

if __name__ == "__main__":
    main()
