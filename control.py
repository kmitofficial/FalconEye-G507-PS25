# rover_control.py
import subprocess

class RoverController:
    def __init__(self, exe_path="./rover_controller"):
        self.proc = subprocess.Popen(
            [exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        print("✅ Rover controller started")

    def send_bbox(self, bbox):
        """
        bbox: (x, y, w, h) or None
        returns controller response string
        """
        if bbox is None:
            self.proc.stdin.write("0 0 0 0\n")
            self.proc.stdin.flush()
            return "STOP"

        x, y, w, h = bbox
        self.proc.stdin.write(f"{x} {y} {w} {h}\n")
        self.proc.stdin.flush()

        return self.proc.stdout.readline().strip()

    def stop(self):
        if self.proc:
            self.proc.stdin.write("0 0 0 0\n")
            self.proc.stdin.flush()

    def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
            print("🛑 Rover stopped")
