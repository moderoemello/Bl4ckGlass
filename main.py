import json
import os
import subprocess
import time
import pyautogui
import pytesseract
from PIL import Image
import cv2
import numpy as np

from agents.monitor import monitor_logs  # 🧠 NEW: Monitor agent

# === Load Config ===
with open("config.json") as f:
    config = json.load(f)

chrome_profile = config["chrome_profile"]
resume_path = config["resume_path"]
job_keywords = config["job_keywords"]
job_location = config["job_location"]
email = config["login"]["email"]
password = config["login"]["password"]

# === Logging ===
def log_event(message):
    with open("log.txt", "a") as log_file:
        log_file.write(message + "\n")
    print(message)

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

    if screen is None or template is None:
        log_event("[-] ERROR: Could not load image or template.")
        return None

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
    try:
        launch_chrome()
        log_event("[+] Chrome launched")

        time.sleep(10)  # wait for LinkedIn to load

        screen_path = capture_screen()
        log_event("[+] Screen captured")

        button_path = "assets/easy_apply.png"
        coords = find_element(button_path, screen_path)

        if coords:
            log_event(f"[+] Found Easy Apply at {coords}")
            click(*coords)
        else:
            log_event("[-] Easy Apply not found")
            monitor_logs()  #  Ask LLM for troubleshooting
    except Exception as e:
        log_event(f"[ERROR] Exception occurred: {e}")
        monitor_logs()

if __name__ == "__main__":
    main()
