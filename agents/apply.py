from agents.shared.utils import screenshot
from agents.llm_agent import ask_llm
from PIL import Image
import pytesseract

def apply_to_job(resume_path):
    screenshot("form.png")

    image = Image.open("form.png")
    visible_text = pytesseract.image_to_string(image)

    prompt = (
        "You are helping automate a LinkedIn job application.\n"
        "Here is a screenshot OCR dump from the form:\n\n"
        f"{visible_text}\n\n"
        "Return JSON like: [{\"label\": \"Phone Number\", \"value\": \"555-123-4567\"}]. "
        "Use realistic dummy data for values."
    )

    response = ask_llm(prompt)
    print("\n[LLM Response]")
    print(response)

    # Next phase: parse JSON and fill form fields
