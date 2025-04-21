import subprocess
import time
import sys
import os
import psutil

def is_script_running(script_name):
    for proc in psutil.process_iter(['name', 'cmdline']):
        try:
            if proc.info['cmdline'] and script_name in ' '.join(proc.info['cmdline']):
                if not any('script_guard.py' in arg for arg in proc.info['cmdline']):
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def start_script(script_path):
    return subprocess.Popen([sys.executable, script_path])

def monitor_script(target_script_path, check_interval=10, restart_delay=5):
    script_name = os.path.basename(target_script_path)
    process = None
    print(f"monitering: {target_script_path}")
    try:
        while True:
            if not is_script_running(script_name):
                print(f"target {script_name} not running, waiting for restart...")
                
                if restart_delay > 0:
                    print(f"wait {restart_delay} seconds...")
                    time.sleep(restart_delay)
                
                print(f"starting: {script_name}")
                process = start_script(target_script_path)
                print(f"PID: {process.pid}")
            else:
                print(f"target {script_name} running")
            
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        if process:
            try:
                print("terminating target script...")
                process.terminate()
            except Exception as e:
                print(f"{e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python monitor_script.py <target_script_path> [check interval] [restart delay]")
        sys.exit(1)
    
    target_script = sys.argv[1]
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    delay = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    monitor_script(target_script, interval, delay)