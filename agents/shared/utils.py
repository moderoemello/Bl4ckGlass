import time
import pyautogui

def screenshot(path="screen.png"):
    pyautogui.screenshot(path)
    return path

def wait(seconds=1):
    time.sleep(seconds)
