import os
import subprocess
import sys
import time
import webbrowser
from threading import Thread

def run_backend():
    """Run the FastAPI backend server."""
    print("Starting FastAPI backend server...")
    if sys.platform.startswith('win'):
        subprocess.Popen(["python", "app.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen(["python", "app.py"])

def run_frontend():
    """Run the React frontend development server."""
    print("Starting React frontend development server...")
    os.chdir("frontend")
    if sys.platform.startswith('win'):
        subprocess.Popen(["npm", "start"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        subprocess.Popen(["npm", "start"])

def open_browser():
    """Open the browser to the frontend URL after a delay."""
    time.sleep(5)  # Wait for servers to start
    print("Opening browser...")
    webbrowser.open("http://localhost:3000")

if __name__ == "__main__":
    # Start the backend
    backend_thread = Thread(target=run_backend)
    backend_thread.daemon = True
    backend_thread.start()
    
    # Start the frontend
    frontend_thread = Thread(target=run_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()
    
    # Open the browser
    browser_thread = Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("Press Ctrl+C to stop the servers...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down servers...")
        sys.exit(0) 