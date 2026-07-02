import time
import os
import csv
import psutil
import pynvml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter


def export_csv_to_excel(csv_filename, excel_filename):
    """
    Read the .csv file and converts it to excel
    """
    print(f"\nGenerating Excel file.")
    try:
        df_export = pd.read_csv(csv_filename)
        df_export.to_excel(excel_filename, index=False, engine='openpyxl')
        print(f"Excel file saved with success as: {excel_filename}")
    except Exception as e:
        print(f"Error during the creation of the Excel file: {e}")


def generate_benchmark_plot(csv_filename, plot_filename):
    """
    Reads the .csv file and creates the benchmark plot.
    """
    print(f"Generating the graph.")
    try:
        df = pd.read_csv(csv_filename)
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.plot(df['Time'], df['FastAPI_CPU_%'], label='CPU (%)', color='blue', linewidth=2)
        ax1.plot(df['Time'], df['GPU_Util_%'], label='GPU (%)', color='green', linewidth=2)
        ax1.set_title('Using of CPU and GPU for computation')
        ax1.set_ylabel('%')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

        ax2.plot(df['Time'], df['FastAPI_RAM_MB'], label='RAM (MB)', color='orange', linewidth=2)
        ax2.plot(df['Time'], df['GPU_VRAM_MB'], label='VRAM (MB)', color='red', linewidth=2)
        ax2.set_title('Memory Impact')
        ax2.set_ylabel('Megabytes (MB)')
        ax2.set_xlabel('Time')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

        ax2.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300)
        print(f"Image saved with success as: {plot_filename}")

    except Exception as e:
        print(f"Error generating the image: {e}")


# init NVIDIA driver
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

SERVER_PORT = 8080
NUM_CORES = float(os.environ.get('POD_VCPU_LIMIT', 1))

found_pid = None
for conn in psutil.net_connections(kind='inet'):
    # check if the process found is listening the server port
    if conn.laddr.port == SERVER_PORT and conn.status == 'LISTEN' and conn.pid:
        found_pid = conn.pid
        break

if not found_pid:
    print(f"No process listening on port {SERVER_PORT}.")
    print("The server must be running.")
    exit(1)

p = psutil.Process(found_pid)

parent = p.parent()

# standard names for master processes
server_names = ['python', 'gunicorn', 'uvicorn']

if len(p.children()) > 0:
    # if the process has children is the master
    main_process = p

elif parent is not None and any(name in parent.name().lower() for name in server_names):
    # the process doesn't have children and the parent has a standard names -> it is one of the worker
    main_process = parent

else:
    # the process doesn't have children and the parent is bash/zsh/systemd -> it is a single worker (--workers 1)
    main_process = p

fastapi_pid = main_process.pid
print(
    f"Monitoring started. Tracking True Master PID: {fastapi_pid} (Port {SERVER_PORT} Num. Cores {NUM_CORES}) and all its workers.")
# ---------------------------------------------------

tracked_processes = {}
initial_processes = [main_process] + main_process.children(recursive=True)

for p in initial_processes:
    tracked_processes[p.pid] = p
    try:
        # cpu percentage, the first time the function is called it returns 0, then the real value
        p.cpu_percent(interval=None)
    except psutil.NoSuchProcess:
        pass

with open('hardware_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time', 'FastAPI_CPU_%', 'FastAPI_RAM_MB', 'GPU_Util_%', 'GPU_VRAM_MB'])

    try:
        while True:
            try:
                # checks if there are new children and adds the pid
                children = main_process.children(recursive=True)
                current_pids = [main_process.pid] + [p.pid for p in children]
            except psutil.NoSuchProcess:
                print("\nFastAPI main server closed. Ending monitoring.")
                break

            total_cpu_util = 0.0
            total_ram_mb = 0.0

            print("-" * 30)

            for p in children:
                if p.pid not in tracked_processes:
                    tracked_processes[p.pid] = p
                    try:
                        p.cpu_percent(interval=None)
                    except psutil.NoSuchProcess:
                        pass

            for pid in list(tracked_processes.keys()):
                if pid in current_pids:
                    try:
                        proc = tracked_processes[pid]

                        # %cpu for that single worker
                        cpu_single_worker = proc.cpu_percent(interval=None)

                        process_type = "Master" if pid == fastapi_pid else "Worker"
                        print(f"{process_type} PID {pid}: {cpu_single_worker:.1f}% CPU")

                        # adds all the %cpu
                        total_cpu_util += cpu_single_worker
                        total_ram_mb += proc.memory_info().rss / (1024 * 1024)
                    except psutil.NoSuchProcess:
                        pass
                else:
                    del tracked_processes[pid]

            normalized_cpu_util = total_cpu_util / NUM_CORES

            # monitoring GPU
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
            except pynvml.NVMLError:
                gpu_util = 0

            try:
                vram_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024 * 1024)
            except pynvml.NVMLError:
                vram_mb = 0

            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            writer.writerow([timestamp, normalized_cpu_util, total_ram_mb, gpu_util, vram_mb])
            f.flush()

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nMonitoring ended. Data saved in hardware_metrics.csv")

        export_csv_to_excel('hardware_metrics.csv', 'hardware_metrics.xlsx')
        generate_benchmark_plot('hardware_metrics.csv', 'benchmark_report.png')