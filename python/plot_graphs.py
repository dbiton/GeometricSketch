import subprocess


tester_path = "build/DynamicSketch"
pcap_path = "pcaps/file.out"
args = f"--file {pcap_path} --type dynamic --repeat 100 log_accuracy"
subprocess.call(f"{tester_path} {args}", shell=True)
