import pyautogui
import time

def click(x, y):
    pyautogui.moveTo(x, y, duration=0.5)
    pyautogui.click()

def type_text(text):
    pyautogui.write(text, interval=0.05)

def press_enter():
    pyautogui.press("enter")

def safe_click(template_path, vision_fn, screenshot_fn):
    screenshot_fn()
    coords = vision_fn(template_path)
    if coords:
        click(*coords)
        return True
    return False
