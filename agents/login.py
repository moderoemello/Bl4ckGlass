from agents.shared.utils import wait, screenshot
from agents.shared.vision import find_element
from agents.shared.input import click, type_text, safe_click

def perform_login(email, password):
    wait(5)
    screenshot()

    if safe_click("assets/email_field.png", find_element, screenshot):
        type_text(email)
    wait(1)

    if safe_click("assets/password_field.png", find_element, screenshot):
        type_text(password)
    wait(1)

    safe_click("assets/sign_in.png", find_element, screenshot)
