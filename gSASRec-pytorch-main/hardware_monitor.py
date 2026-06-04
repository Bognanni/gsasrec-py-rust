import time
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

        ax1.plot(df['Time'], df['CPU_Total_%'], label='CPU (%)', color='blue', linewidth=2)
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

SERVER_PORT = 8081
fastapi_pid = None

# process linked to the port
for conn in psutil.net_connections(kind='inet'):
    if conn.laddr.port == SERVER_PORT and conn.status == 'LISTEN':
        fastapi_pid = conn.pid
        break

if not fastapi_pid:
    print(f"No process listening on the {SERVER_PORT}.")
    print("The server must be running.")
    exit(1)

process = psutil.Process(fastapi_pid)
print(f"Monitoring started. Tracking PID FastAPI: {fastapi_pid} (Port {SERVER_PORT})")

with open('hardware_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time', 'CPU_Total_%', 'FastAPI_RAM_MB', 'GPU_Util_%', 'GPU_VRAM_MB'])

    try:
        while True:
            cpu_util = psutil.cpu_percent()

            try:
                ram_mb = process.memory_info().rss / (1024 * 1024)
            except psutil.NoSuchProcess:
                print("\nFastAPI server closed. Ending monitoring.")
                break

            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
            except pynvml.NVMLError:
                gpu_util = 0

            try:
                vram_mb = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024 * 1024)
            except pynvml.NVMLError:
                vram_mb = 0

            timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            writer.writerow([timestamp, cpu_util, ram_mb, gpu_util, vram_mb])
            f.flush()

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nMonitoring ended. Data saved in hardware_metrics.csv")

        export_csv_to_excel('hardware_metrics.csv', 'hardware_metrics.xlsx')

        generate_benchmark_plot('hardware_metrics.csv', 'benchmark_report.png')