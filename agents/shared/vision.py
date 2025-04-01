import cv2

def find_element(template_path, screenshot_path="screen.png", threshold=0.75):
    screen = cv2.imread(screenshot_path)
    template = cv2.imread(template_path)
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        h, w = template.shape[:2]
        return max_loc[0] + w // 2, max_loc[1] + h // 2
    return None
