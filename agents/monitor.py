import os
from agents.llm_agent import ask_llm

LOG_PATH = "log.txt"

def monitor_logs():
    if not os.path.exists(LOG_PATH):
        print("[Monitor] No logs found.")
        return

    with open(LOG_PATH, "r") as f:
        logs = f.read()

    prompt = (
        "You are monitoring an automation tool for LinkedIn job applications.\n"
        "Here is the log output:\n\n"
        f"{logs}\n\n"
        "Identify what errors occurred and suggest helpful advice or next steps for the user.\n"
        "Return clear instructions or diagnostic questions."
    )

    response = ask_llm(prompt)
    print("\n[Monitor Agent Suggestion]")
    print(response)

