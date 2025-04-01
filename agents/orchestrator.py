import subprocess
from agents.login import perform_login
from agents.apply import apply_to_job
from agents.shared.utils import wait

def launch_chrome(profile):
    subprocess.Popen([
        "google-chrome",
        f"--profile-directory={profile}",
        "--new-window",
        "https://www.linkedin.com/login"
    ])
    wait(5)

def run_workflow(config):
    launch_chrome(config["chrome_profile"])
    perform_login(config["login"]["email"], config["login"]["password"])
    apply_to_job(config["resume_path"])
