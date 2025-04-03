import os
import time
import random
import logging
import json
import subprocess
import pytesseract
from pytesseract import Output
import pyautogui
from PIL import Image
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletion
import base64
import layoutparser as lp
from screeninfo import get_monitors
from io import BytesIO
import mss
import numpy as np
import re
from typing import Tuple, Optional
import cv2  # OpenCV for image matching
try:
    import pyautogui  # For screenshots and screen size; ensure it's installed
except ImportError:
    pyautogui = None

# Configure logging for the system
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load configuration (credentials, search parameters, etc.)
CONFIG = {}
try:
    # Try loading from a JSON config file
    import json
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
        logging.info("Configuration loaded from config.json")
except Exception as e:
    logging.warning("Could not load config file, using environment variables/defaults. (%s)", e)
    # Load from environment variables as fallback
    CONFIG["username"] = os.getenv("LINKEDIN_USERNAME", "<your_email>")
    CONFIG["password"] = os.getenv("LINKEDIN_PASSWORD", "<your_password>")
    CONFIG["job_keywords"] = os.getenv("JOB_KEYWORDS", "Software Engineer")
    CONFIG["job_location"] = os.getenv("JOB_LOCATION", "United States")
    CONFIG["resume_file_path"] = os.getenv("RESUME_PATH", "/path/to/resume.pdf")
    # Additional config options (like filters or experience level) can be added here.
    logging.info("Configuration loaded from environment/defaults.")

# Ensure pyautogui failsafe is enabled (moving mouse to corner aborts script)
pyautogui.FAILSAFE = True










def get_monitors_info():
    """Detect monitor resolutions and layout offsets for the current system."""
    info = []
    system = platform.system()
    if system == "Linux":
        # Use xrandr to get connected monitor info
        try:
            output = subprocess.check_output(["xrandr", "--query"]).decode("utf-8")
        except Exception as e:
            raise RuntimeError("Failed to run xrandr: " + str(e))
        # Parse lines like: "HDMI-1 connected primary 1920x1080+0+0"
        for line in output.splitlines():
            if " connected" in line:
                # Extract name and geometry
                parts = line.split()
                name = parts[0]
                geom_match = re.search(r"(\d+)x(\d+)\+(\d+)\+(\d+)", line)
                if geom_match:
                    width = int(geom_match.group(1))
                    height = int(geom_match.group(2))
                    offset_x = int(geom_match.group(3))
                    offset_y = int(geom_match.group(4))
                    # Check if this monitor is primary
                    primary = "primary" in line
                    info.append({
                        "name": name, "width": width, "height": height,
                        "x": offset_x, "y": offset_y, "primary": primary
                    })
    elif system == "Windows":
        # Use ctypes to get all monitor dimensions and positions
        try:
            import ctypes
        except ImportError:
            raise RuntimeError("ctypes not available on Windows for monitor detection.")
        user32 = ctypes.windll.user32
        monitors = []
        MONITOR_ENUM_PROC = ctypes.WINFUNCTYPE(
            ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.POINTER(ctypes.c_long * 4), ctypes.c_double
        )
        def callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
            rct = lprcMonitor.contents
            # rct is a ctypes array [left, top, right, bottom]
            left, top, right, bottom = int(rct[0]), int(rct[1]), int(rct[2]), int(rct[3])
            monitors.append((left, top, right, bottom))
            return 1
        # EnumDisplayMonitors(0,0,callback,0) will fill monitors list
        user32.EnumDisplayMonitors(0, 0, MONITOR_ENUM_PROC(callback), 0)
        for (left, top, right, bottom) in monitors:
            info.append({
                "x": left, "y": top,
                "width": right - left,
                "height": bottom - top,
                "primary": (left == 0 and top == 0)  # simple check for primary
            })
    elif system == "Darwin":
        # macOS: try using pyautogui (which uses Quartz) as a fallback for primary monitor
        if pyautogui:
            w, h = pyautogui.size()
        else:
            # As a last resort, use a default command (requires Quartz or system_profiler)
            try:
                output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"]).decode("utf-8")
            except Exception as e:
                raise RuntimeError("Failed to get display info on macOS: " + str(e))
            # Parse the resolution from system_profiler output if possible
            # (Simplified parsing: look for resolution lines)
            m = re.search(r"Resolution: (\d+) x (\d+)", output)
            if m:
                w, h = int(m.group(1)), int(m.group(2))
            else:
                w, h = None, None
        if w and h:
            info.append({"x": 0, "y": 0, "width": w, "height": h, "primary": True})
    else:
        # Other OS: try pyautogui or fallback to  single-monitor assumption
        if pyautogui:
            w, h = pyautogui.size()
            info.append({"x": 0, "y": 0, "width": w, "height": h, "primary": True})
    return info

def get_window_geometry(window_title: Optional[str] = None):
    """Get the active or specified window's absolute position and content size on Linux using xdotool/xwininfo."""
    system = platform.system()
    if system != "Linux":
        return None  # window geometry retrieval is implemented for Linux only in this example
    # Get window ID
    try:
        if window_title:
            # Find window by title (uses the first match)
            win_id = subprocess.check_output(
                ["xdotool", "search", "--name", window_title]
            ).decode("utf-8").strip().split('\n')[0]
        else:
            # Get active window
            win_id = subprocess.check_output(["xdotool", "getactivewindow"]).decode("utf-8").strip()
    except Exception as e:
        raise RuntimeError("Failed to get window ID via xdotool: " + str(e))
    # Get window geometry via xwininfo
    try:
        xwininfo_out = subprocess.check_output(["xwininfo", "-id", win_id]).decode("utf-8")
    except Exception as e:
        raise RuntimeError("Failed to get window geometry via xwininfo: " + str(e))
    # Parse Absolute position and Width/Height from xwininfo output
    geom = {}
    for line in xwininfo_out.splitlines():
        if "Absolute upper-left X:" in line:
            geom["abs_x"] = int(line.split(":")[1])
        elif "Absolute upper-left Y:" in line:
            geom["abs_y"] = int(line.split(":")[1])
        elif "Width:" in line:
            geom["width"] = int(line.split(":")[1])
        elif "Height:" in line:
            geom["height"] = int(line.split(":")[1])
    # Get window frame extents (borders/titlebar) via xprop
    try:
        # This returns a line like: _NET_FRAME_EXTENTS(CARDINAL) = left, right, top, bottom
        xprop_out = subprocess.check_output(["xprop", "-id", win_id, "_NET_FRAME_EXTENTS"]).decode("utf-8")
        # Extract the four numbers
        extents = re.findall(r"\d+", xprop_out)
        if len(extents) >= 4:
            left, right, top, bottom = map(int, extents[:4])
        else:
            left = right = top = bottom = 0
    except Exception:
        # If xprop or property not available, assume no decoration offsets
        left = right = top = bottom = 0
    # Adjust absolute coordinates to include frame extents (get outer window origin)
    abs_x = geom.get("abs_x", 0) - left
    abs_y = geom.get("abs_y", 0) - top
    width = geom.get("width", 0) + left + right
    height = geom.get("height", 0) + top + bottom
    return {"x": abs_x, "y": abs_y, "width": width, "height": height}

def refine_coordinates(approx_x: int, approx_y: int, window_title: Optional[str] = None,
                       template_path: Optional[str] = None) -> Tuple[int, int]:
    """
    Refine the (approx_x, approx_y) coordinates of a UI element to actual screen coordinates.
    If window_title is provided, restrict search to that window. Optionally use a template image for matching.
    Returns corrected (x, y) coordinates.
    """
    # 1. Get monitors info
    monitors = get_monitors_info()
    # 2. Determine base coordinates (if window context is available)
    base_x = approx_x
    base_y = approx_y
    window_geom = None
    if window_title or platform.system() == "Linux":
        # Only attempt window geometry on Linux (for Windows/macOS, one could use other methods or skip if not needed)
        try:
            window_geom = get_window_geometry(window_title)
        except RuntimeError as e:
            print(f"Warning: {e}", file=sys.stderr)
            window_geom = None
    if window_geom:
        # If we have window content geometry, treat approx coordinates as relative to content
        base_x = window_geom["x"] + approx_x
        base_y = window_geom["y"] + approx_y
        # Identify which monitor contains this window (by checking window center)
        win_cx = base_x + window_geom["width"]//2
        win_cy = base_y + window_geom["height"]//2
        for mon in monitors:
            if win_cx >= mon["x"] and win_cx < mon["x"] + mon["width"] and \
               win_cy >= mon["y"] and win_cy < mon["y"] + mon["height"]:
                # Found the monitor containing the window
                # (No additional offset needed since window_geom.x already absolute)
                break
    else:
        # No window info: assume approx_x, approx_y might already be absolute or relative to primary
        # If monitors > 1, check if coords fall in any monitor region; if not, default to primary
        found_mon = None
        for mon in monitors:
            if approx_x >= mon["x"] and approx_x < mon["x"] + mon["width"] and \
               approx_y >= mon["y"] and approx_y < mon["y"] + mon["height"]:
                found_mon = mon
                break
        if found_mon:
            base_x = approx_x  # already in global coords for that monitor
            base_y = approx_y
        else:
            # If coordinate was given relative to some monitor (e.g., second) but we can't tell, assume primary for now
            primary_mon = next((m for m in monitors if m.get("primary")), None)
            if primary_mon:
                base_x = primary_mon["x"] + approx_x
                base_y = primary_mon["y"] + approx_y
            else:
                base_x = approx_x
                base_y = approx_y
    # 3. Screenshot the region of interest for template matching
    screenshot_img = None
    if pyautogui:
        try:
            if window_geom:
                # Capture the window content region (use width/height of content area)
                screenshot_img = pyautogui.screenshot(region=(window_geom["x"], window_geom["y"],
                                                             window_geom["width"], window_geom["height"]))
            else:
                # Capture entire screen(s)
                screenshot_img = pyautogui.screenshot()
        except Exception as e:
            print(f"Warning: PyAutoGUI screenshot failed: {e}", file=sys.stderr)
    if screenshot_img is None:
        # Fallback: try mss if pyautogui not available or failed
        try:
            from mss import mss
            with mss() as sct:
                if window_geom:
                    mon_region = {
                        "top": window_geom["y"], "left": window_geom["x"],
                        "width": window_geom["width"], "height": window_geom["height"]
                    }
                    sct_img = sct.grab(mon_region)
                else:
                    # Grab all monitors
                    sct_img = sct.grab(sct.monitors[0])
                # Convert to PIL Image for consistency
                from PIL import Image
                screenshot_img = Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb)
        except Exception as e:
            raise RuntimeError("Failed to capture screen for template matching: " + str(e))
    # Convert screenshot to OpenCV image (numpy array)
    screen_np = np.array(screenshot_img)
    # PyAutoGUI/PIL gives image in RGB order; convert to BGR for OpenCV or directly to gray
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_RGB2GRAY)
    # 4. Template matching to refine coordinates
    match_x, match_y = base_x, base_y  # default to base if matching fails
    if template_path:
        # Use provided template image
        template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template_img is None:
            raise FileNotFoundError(f"Template image not found: {template_path}")
        # Search within the window region if available, otherwise full screen
        search_img = screen_gray
        # Perform template matching
        res = cv2.matchTemplate(search_img, template_img, cv2.TM_CCOEFF_NORMED)  # :contentReference[oaicite:2]{index=2}
        _minVal, maxVal, _minLoc, maxLoc = cv2.minMaxLoc(res)
        if maxVal < 0.5:
            print("Warning: Template match confidence is low (%.2f). Coordinates may be off." % maxVal, file=sys.stderr)
        top_left = maxLoc  # top-left corner of best match
        h_t, w_t = template_img.shape[:2]
        # Compute center of matched region in the screenshot coordinate space
        center_x = top_left[0] + w_t // 2
        center_y = top_left[1] + h_t // 2
        # If we captured only the window region, the screenshot origin = window_geom.x,y
        if window_geom:
            match_x = window_geom["x"] + center_x
            match_y = window_geom["y"] + center_y
        else:
            match_x = center_x
            match_y = center_y
    else:
        # No template provided: use a small patch around the approximate point as template
        # Define a patch around (base_x, base_y) in the screenshot coordinates
        # Convert global base coords to screenshot-local coords:
        local_x = base_x
        local_y = base_y
        if window_geom:
            # If screenshot is window only, window_geom.x,y is origin 0,0 in screenshot
            local_x = base_x - window_geom["x"]
            local_y = base_y - window_geom["y"]
        # Patch size (e.g., 40x40 box centered at approx point, clipped to image bounds)
        patch_size = 40
        half = patch_size // 2
        x0 = max(0, local_x - half)
        y0 = max(0, local_y - half)
        x1 = min(screen_gray.shape[1], local_x + half)
        y1 = min(screen_gray.shape[0], local_y + half)
        template_img = screen_gray[y0:y1, x0:x1]
        # Search area: a slightly larger region around the approximate point
        search_radius = 60
        sx0 = max(0, local_x - search_radius)
        sy0 = max(0, local_y - search_radius)
        sx1 = min(screen_gray.shape[1], local_x + search_radius)
        sy1 = min(screen_gray.shape[0], local_y + search_radius)
        search_img = screen_gray[sy0:sy1, sx0:sx1]
        # Only proceed if the search region is larger than the template
        if search_img.shape[0] >= template_img.shape[0] and search_img.shape[1] >= template_img.shape[1]:
            res = cv2.matchTemplate(search_img, template_img, cv2.TM_CCOEFF_NORMED)
            _minVal, maxVal, _minLoc, maxLoc = cv2.minMaxLoc(res)
            best_local_x = maxLoc[0]
            best_local_y = maxLoc[1]
            # Convert best match location back to global screenshot coords
            best_global_x = sx0 + best_local_x + template_img.shape[1]//2
            best_global_y = sy0 + best_local_y + template_img.shape[0]//2
            # Map back to full screen coordinates
            if window_geom:
                match_x = window_geom["x"] + best_global_x
                match_y = window_geom["y"] + best_global_y
            else:
                match_x = best_global_x
                match_y = best_global_y
    # 5. Return the refined global coordinates
    return int(match_x), int(match_y)

# Example usage (uncomment and modify accordingly):
# approx_x, approx_y = 100, 200  # coordinates from GPT-4 Vision
# corrected_x, corrected_y = refine_coordinates(approx_x, approx_y, window_title="LinkedIn", template_path="element.png")
# print("Corrected coordinates:", corrected_x, corrected_y)
# pyautogui.click(corrected_x, corrected_y)  # perform the click










class VisionAgent:
    def __init__(self, orchestrator, api_key):
        self.orchestrator = orchestrator
        self.llm_agent = LLMIntegrationAgent(api_key)
        self.layout_model = lp.Detectron2LayoutModel(
            config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            enforce_cpu=True
        )
        self.primary_monitor = get_monitors()[0]
        self.screen_width = self.primary_monitor.width
        self.screen_height = self.primary_monitor.height

    def capture_screen(self, region=None):
        with mss.mss() as sct:
            monitor = {
                "left": region[0] if region else 0,
                "top": region[1] if region else 0,
                "width": region[2] if region else self.screen_width,
                "height": region[3] if region else self.screen_height
            }
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", (sct_img.width, sct_img.height), sct_img.rgb)
            return img, sct_img.width, sct_img.height

    def read_text(self, region=None):
        image, _, _ = self.capture_screen(region)
        return pytesseract.image_to_string(image)

    def find_text_position(self, target_text, region=None):
        image, _, _ = self.capture_screen(region)
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        for i, word in enumerate(data["text"]):
            if word and target_text.lower() in word.lower():
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                return (x + w // 2, y + h // 2)
        return None

    def find_text_with_layoutparser(self, label="Title"):
        image, _, _ = self.capture_screen()
        image_np = np.array(image)
        layout = self.layout_model.detect(image_np)
        results = []
        for block in layout:
            if block.type == label:
                x, y = int(block.block.x_1 + (block.block.width / 2)), int(block.block.y_1 + (block.block.height / 2))
                results.append((label, (x, y)))
        return results

    def match_template(self, template_path, threshold=0.8):
        screen_img, _, _ = self.capture_screen()
        screen_np = np.array(screen_img.convert("RGB"))
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        res = cv2.matchTemplate(screen_np, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        matches = []
        for pt in zip(*loc[::-1]):
            x = int(pt[0] + template.shape[1] / 2)
            y = int(pt[1] + template.shape[0] / 2)
            matches.append(("TemplateMatch", (x, y)))
        return matches

    def analyze_screenshot_with_gpt(self, prompt):
        image, screen_w, screen_h = self.capture_screen()
        gpt_result = self.llm_agent.analyze_image_with_gpt(image, prompt)
        return gpt_result, screen_w, screen_h

    def get_click_coordinates_from_all_methods(self, prompt):
        img_pil, screen_w, screen_h = self.capture_screen()
        img_np = np.array(img_pil)

        # GPT Vision-based
        try:
            result_text = self.llm_agent.analyze_image_with_gpt(img_pil, prompt)
            gpt_points = json.loads(result_text)
        except Exception as e:
            logging.error("GPT output JSON error: %s", e)
            gpt_points = []

        img_w, img_h = img_pil.size
        scale_x = screen_w / img_w
        scale_y = screen_h / img_h

        final_coords = []

        for item in gpt_points:
            if all(k in item for k in ("title", "x", "y")):
                corrected_x = int(item["x"] * scale_x)
                corrected_y = int(item["y"] * scale_y)
                final_coords.append((item["title"], (corrected_x, corrected_y)))
                logging.debug(f"[GPT] {item['title']}: {corrected_x}, {corrected_y}")

        # Add OCR fallback
        ocr_coord = self.find_text_position(prompt)
        if ocr_coord:
            final_coords.append(("OCR_Fallback", ocr_coord))
            logging.debug(f"[OCR] Found '{prompt}' at {ocr_coord}")

        # Add LayoutParser fallback
        layout_coords = self.find_text_with_layoutparser()
        final_coords.extend(layout_coords)

        # Template matching fallback
        template_matches = self.match_template(f"templates/{prompt}.png")
        final_coords.extend(template_matches)

        # Draw debug overlay
        try:
            draw = ImageDraw.Draw(img_pil)
            for label, (x, y) in final_coords:
                draw.rectangle([(x - 10, y - 10), (x + 10, y + 10)], outline="red", width=2)
                draw.text((x + 12, y), label, fill="white")
            img_pil.save("debug_overlay.png")
            logging.info("Saved debug image with overlay to debug_overlay.png")
        except Exception as e:
            logging.warning("Could not create overlay debug image: %s", e)

        return final_coords



def debug_overlay_image(self, image, coords):
    draw = ImageDraw.Draw(image)
    for title, (x, y) in coords:
        draw.rectangle((x-5, y-5, x+5, y+5), outline="red", width=2)
        draw.text((x+10, y), title, fill="white")
    image.save("debug_overlay.png")



class InputAgent:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.mouse_speed_range = (0.2, 0.5)
        self.mouse_step_pixels = 10
        self.typing_delay_range = (0.05, 0.15)

    def move_mouse(self, target_x, target_y):
        import math
        start_x, start_y = pyautogui.position()
        distance = ((target_x - start_x) ** 2 + (target_y - start_y) ** 2) ** 0.5
        steps = max(3, int(distance / self.mouse_step_pixels))
        for i in range(1, steps + 1):
            t = i / float(steps)
            ease_t = (1 - math.cos(t * math.pi)) / 2
            interp_x = start_x + (target_x - start_x) * ease_t
            interp_y = start_y + (target_y - start_y) * ease_t
            jitter_px = max(1, int(distance * 0.005))
            interp_x += random.randint(-jitter_px, jitter_px) * (1 - t)
            interp_y += random.randint(-jitter_px, jitter_px) * (1 - t)
            pyautogui.moveTo(int(interp_x), int(interp_y), duration=0.01)
        pyautogui.moveTo(target_x, target_y, duration=0.05)

    def click(self, x=None, y=None, button='left'):
        if x is not None and y is not None:
            self.move_mouse(x, y)
        time.sleep(random.uniform(0.05, 0.3))
        pyautogui.click(button=button)
        logging.debug("Clicked at (%s, %s) with button %s", str(x), str(y), button)

    def type_text(self, text):
        for char in text:
            pyautogui.write(char)
            time.sleep(random.uniform(*self.typing_delay_range))
        logging.debug("Typed text: %s", text)

    def press_key(self, key):
        pyautogui.press(key)
        logging.debug("Pressed key: %s", key)

    def hotkey(self, *keys):
        pyautogui.hotkey(*keys)
        logging.debug("Pressed hotkey combination: %s", "+".join(keys))

class LoginAgent:
    """Agent to handle LinkedIn login using the provided credentials."""
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.vision = orchestrator.vision
        self.input = orchestrator.input
    
    def perform_login(self):
        """Open LinkedIn job search page directly using logged-in Chrome profile."""
        logging.info("Launching LinkedIn job search page with user profile (assumes already logged in).")

        profile = CONFIG.get("chrome_profile", "Default")
        chrome_path = "/usr/bin/google-chrome"

        # Build job search URL
        base_url = "https://www.linkedin.com/jobs/search?"
        query_params = []
        if CONFIG.get("job_keywords"):
            query_params.append("keywords=" + CONFIG["job_keywords"].replace(" ", "%20"))
        if CONFIG.get("job_location"):
            query_params.append("location=" + CONFIG["job_location"].replace(" ", "%20"))
        query_params.append("f_LF=f_AL")  # Easy Apply filter
        jobs_url = base_url + "&".join(query_params)

        chrome_args = [
            chrome_path,
            f"--profile-directory={profile}",
            "--new-window",
            jobs_url
        ]

        try:
            subprocess.Popen(chrome_args)
            logging.info("Chrome launched with profile: %s", profile)
        except Exception as e:
            logging.error("Failed to launch Chrome: %s", e)
            return

        time.sleep(5)
        self.orchestrator.context["logged_in"] = True

            



class SearchAgent:
    """Agent to navigate to the Jobs page, enter search criteria, and filter results."""
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.vision = orchestrator.vision
        self.input = orchestrator.input
    
    def open_jobs_page(self):
        """Navigate to LinkedIn Jobs search page with given filters."""
        logging.info("Navigating to LinkedIn Jobs search page...")
        # Construct the jobs search URL with query and location. We also filter for Easy Apply jobs if possible.
        base_url = "https://www.linkedin.com/jobs/search?"
        query_params = []
        if CONFIG.get("job_keywords"):
            query_params.append("keywords=" + CONFIG["job_keywords"].replace(" ", "%20"))
        if CONFIG.get("job_location"):
            query_params.append("location=" + CONFIG["job_location"].replace(" ", "%20"))
        # LinkedIn URL param for Easy Apply filter (if known). Using 'f_AL=true' as placeholder for Easy Apply.
        query_params.append("f_LF=f_AL")  # This might not exactly filter Easy Apply; included as an example.
        jobs_url = base_url + "&".join(query_params)
        # Open the jobs search URL in a new browser tab
        self.input.hotkey('ctrl', 't')  # new tab
        time.sleep(0.5)
        pyautogui.write(jobs_url)
        self.input.press_key('enter')
        # Give time for the page to load results
        time.sleep(5)
    
    def apply_filters(self):
        """Apply Easy Apply filter via UI if not already filtered by URL (optional step)."""
        # Try to find "Easy Apply" filter toggle on the page and click it
        filter_button = self.vision.find_text_position("Easy Apply")
        if filter_button:
            logging.info("Applying 'Easy Apply' filter.")
            self.input.click(filter_button[0], filter_button[1])
            time.sleep(2)
        else:
            logging.info("'Easy Apply' filter option not found on screen (it might already be applied or named differently).")
    
    def get_job_listings(self):
        """
        Use GPT-4 Vision to analyze the LinkedIn jobs page and return job title + adjusted coordinates.
        """
        logging.info("Capturing screen to identify job listings using GPT-4 Vision...")
        prompt = (
            "You're looking at a LinkedIn job search results page. Extract up to 10 job listings visible in the left column. "
            "Return ONLY valid JSON with this format:\n"
            "[{\"title\": \"Job Title\", \"x\": 123, \"y\": 456}]\n"
            "No explanation or additional formatting—just valid JSON."
        )
        job_coords = self.vision.get_click_coordinates_from_all_methods(prompt)
        logging.info("Parsed %d job listings from GPT-4.", len(job_coords))
        return job_coords

        try:
            import json
            parsed = json.loads(result_text)
            scale_x = screen_width / img_width
            scale_y = screen_height / img_height

            job_coords = []
            for item in parsed:
                if "title" in item and "x" in item and "y" in item:
                    adj_x = int(item["x"] * scale_x)
                    adj_y = int(item["y"] * scale_y)
                    job_coords.append((item["title"], (adj_x, adj_y)))

            logging.info("Parsed %d job listings from GPT-4.", len(job_coords))
            return job_coords

        except Exception as e:
            logging.error("Failed to parse GPT-4 response: %s", e)
            return []

    def select_job(self, job_coord):
        """Click a job listing given its coordinates to open the job detail pane."""
        title, (x, y) = job_coord
        logging.info("Selecting job listing: %s", title)
        self.input.click(x, y)
        # Wait for job details pane to load (detected by presence of "Easy Apply" or "Apply" button or job title in detail pane)
        time.sleep(3)

class ApplicationAgent:
    """Agent to handle the Easy Apply process for a selected job."""
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.vision = orchestrator.vision
        self.input = orchestrator.input
    
    def apply_to_job(self):
        """Execute the Easy Apply workflow on the currently open job detail pane."""
        # Check for "Easy Apply" or "Apply" button in the job detail view
        apply_btn = self.vision.find_text_position("Easy Apply") or self.vision.find_text_position("Apply now")
        if not apply_btn:
            logging.info("No Easy Apply option for this job, skipping application.")
            return False
        # Click the Easy Apply button
        logging.info("Easy Apply button found. Clicking to start application.")
        self.input.click(apply_btn[0], apply_btn[1])
        time.sleep(3)  # Wait for the application modal to appear
        
        # Loop through application steps
        step = 1
        while True:
            # If a "Next" button is present, we haven't reached final submission yet
            next_btn = self.vision.find_text_position("Next")
            submit_btn = self.vision.find_text_position("Submit") or self.vision.find_text_position("Submit application")
            review_btn = self.vision.find_text_position("Review")  # sometimes there's a Review step
            # Fill in any text fields that are visible on this step:
            screen_text = self.vision.read_text()
            if screen_text:
                # Example: if phone number is requested
                if "Phone" in screen_text or "phone" in screen_text:
                    phone = CONFIG.get("phone_number", "")
                    if phone:
                        field_pos = self.vision.find_text_position("Phone")  # find "Phone" label
                        if field_pos:
                            # Click slightly to the right of the label to focus the input (assuming input is next to label)
                            self.input.click(field_pos[0] + 100, field_pos[1])
                        self.input.type_text(phone)
                        logging.info("Filled phone number.")
                # Additional field handling (email, address, etc.) can be added similarly using CONFIG data.
            # If there's an upload button (for resume or other file)
            upload_btn = self.vision.find_text_position("Upload") or self.vision.find_text_position("Attach")
            if upload_btn:
                logging.info("Upload button found, attempting to attach file.")
                self.input.click(upload_btn[0], upload_btn[1])
                time.sleep(2)
                # Type the resume file path and press enter (assuming file dialog is open)
                self.input.type_text(CONFIG.get("resume_file_path", ""))
                self.input.press_key('enter')
                time.sleep(2)
            
            # Determine which button to click to progress
            if next_btn:
                logging.info("Step %d completed, clicking Next.", step)
                self.input.click(next_btn[0], next_btn[1])
                time.sleep(2)
                step += 1
                continue  # go to next iteration of loop
            elif review_btn:
                logging.info("All fields filled. Clicking Review before final submission.")
                self.input.click(review_btn[0], review_btn[1])
                time.sleep(2)
                # After review, loop will check again for submit
                continue
            elif submit_btn:
                # Final step: click Submit
                logging.info("Final step reached. Clicking Submit to send application.")
                self.input.click(submit_btn[0], submit_btn[1])
                time.sleep(2)
                break  # exit the loop after submitting
            else:
                # Neither Next nor Submit found - could be a confirmation dialog or an issue
                logging.warning("No 'Next' or 'Submit' detected. Possibly an unexpected form or confirmation.")
                # Use GPT-4 to analyze the situation if possible (placeholder)
                analysis = self.vision.analyze_screenshot_with_gpt("Identify if the application is complete or needs input.")
                # Based on analysis (if any), decide what to do - for now, break out.
                break
        
        # After exiting form loop, check for confirmation
        confirmation = self.vision.find_text_position("submitted") or self.vision.find_text_position("Application submitted")
        if confirmation:
            logging.info("Application submitted successfully for this job.")
            return True
        else:
            logging.info("Application process ended, but submission confirmation not detected.")
            return False

class MonitorAgent:
    """Agent to monitor the application process for errors or status updates."""
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.vision = orchestrator.vision
    
    def check_for_errors(self):
        """Scan the screen for any common error messages or issues."""
        text = self.vision.read_text()
        if not text:
            return None
        text_lower = text.lower()
        # Examples of issues to look for:
        if "error" in text_lower or "problem" in text_lower:
            logging.error("Detected an error message on screen: '%s'", text.strip().splitlines()[0])
            return "error"
        if "enter a valid" in text_lower or "required" in text_lower:
            logging.warning("Form is indicating a missing or invalid required field.")
            return "validation_error"
        if "sign in" in text_lower and "linkedin" in text_lower:
            logging.warning("Noticed a sign-in prompt - possibly got logged out.")
            return "logged_out"
        # Could add checks for CAPTCHA, etc.
        return None

class LLMIntegrationAgent:
    """Handles interaction with OpenAI's GPT-4 Turbo Vision API."""
    def __init__(self, api_key):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def analyze_image_with_gpt(self, image, prompt):
        import base64
        from io import BytesIO

        # Convert image to base64-encoded PNG
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{img_str}"

        # Log info
        logging.info("Sending screenshot to GPT with prompt: %s", prompt)
        logging.info("Base64 image size: %d bytes", len(img_str))

        # Compose GPT message
        message_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": message_content}],
                max_tokens=1000,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error("OpenAI API error: %s", e)
            return ""



class Orchestrator:
    """Central orchestrator that initializes agents and controls the flow."""
    def __init__(self):
        self.context = {
            "logged_in": True,
            "jobs_applied": 0,
            "jobs_skipped": 0
        }

        api_key = CONFIG.get("openai_api_key", "sk-...")
        self.vision = VisionAgent(self, api_key)
        self.input = InputAgent(self)
        self.search = SearchAgent(self)
        self.application = ApplicationAgent(self)
        self.monitor = MonitorAgent(self)

    def launch_chrome_with_linkedin(self):
        logging.info("Launching Chrome with LinkedIn Jobs search page...")
        profile = CONFIG.get("chrome_profile", "Default")
        chrome_path = "/usr/bin/google-chrome"  # Adjust if needed

        base_url = "https://www.linkedin.com/jobs/search?"
        query_params = []
        if CONFIG.get("job_keywords"):
            query_params.append("keywords=" + CONFIG["job_keywords"].replace(" ", "%20"))
        if CONFIG.get("job_location"):
            query_params.append("location=" + CONFIG["job_location"].replace(" ", "%20"))
        query_params.append("f_LF=f_AL")
        jobs_url = base_url + "&".join(query_params)

        try:
            subprocess.Popen([
                chrome_path,
                f"--profile-directory={profile}",
                "--new-window",
                jobs_url
            ])
            time.sleep(5)
        except Exception as e:
            logging.error("Failed to launch Chrome: %s", e)

    def run(self):
        logging.info("BL4CKGLASS automation started.")
        logging.info("Assuming already logged in. Skipping login step.")

        self.launch_chrome_with_linkedin()

        self.search.apply_filters()

        job_listings = self.search.get_job_listings()
        if not job_listings:
            logging.error("No job listings found. Exiting.")
            return

        for job in job_listings:
            self.search.select_job(job)

            issue = self.monitor.check_for_errors()
            if issue == "logged_out":
                logging.warning("Detected logout. Manual intervention required.")
                break
            elif issue:
                logging.info("Skipping job due to detected issue: %s", issue)
                self.context["jobs_skipped"] += 1
                continue

            success = self.application.apply_to_job()
            if success:
                self.context["jobs_applied"] += 1
            else:
                self.context["jobs_skipped"] += 1

            sleep_time = random.uniform(3.0, 6.0)
            logging.info("Pausing for %.1f seconds before next job.", sleep_time)
            time.sleep(sleep_time)

        logging.info("Automation complete. Applied: %d | Skipped: %d",
                     self.context["jobs_applied"], self.context["jobs_skipped"])
        logging.info("BL4CKGLASS automation ended.")


# Run the orchestrator if this script is executed
if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run()
