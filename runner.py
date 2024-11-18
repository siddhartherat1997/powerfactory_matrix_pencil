import subprocess
import time
import main

def run_test_and_main():
    # Open a new terminal and run test_run.py
    print("Starting test_run.py in a new terminal...")
    # For Command Prompt (cmd)
    subprocess.Popen(['start', 'cmd', '/k', 'python', 'test_run.py'], shell=True)
    # Alternatively, use PowerShell
    # subprocess.Popen(['start', 'powershell', '-NoExit', '-Command', 'python test_run.py'], shell=True)
    
    # Sleep for a short time to give the new terminal a chance to start
    time.sleep(2)
    
    # Run main.py in the original terminal
    print("Starting main.py...")
    main.main()

if __name__ == "__main__":
    run_test_and_main()

