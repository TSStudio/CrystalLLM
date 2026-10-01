# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""CrystalLLM Monitor."""

import contextlib
import re
import subprocess
import threading
import time
from collections import deque
from enum import Enum, auto
from multiprocessing import Pipe, Process
from pathlib import Path
from threading import Thread

import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for subprocess compatibility
matplotlib.use("Agg")

# Monitoring interval constants (in seconds)
UTILIZATION_INTERVAL = 0.01  # Interval for GPU utilization sampling
MEMORY_INTERVAL = 0.01  # Interval for memory monitoring sampling
NETWORK_INTERVAL = 0.001  # Interval for network bandwidth sampling

# Plot axis limits
UTILIZATION_Y_MAX = 110  # Maximum Y-axis value for utilization plot (%)

# Unit conversion constants
B_TO_GB = 1000_000_000  # Bytes to gigabytes conversion factor
BITS_TO_GIGABITS = 1000_000_000  # Bits to gigabits conversion factor
MILLIWATTS_TO_WATTS = 1000  # Milliwatts to watts conversion factor

# Timeout constants (in seconds)
ETHOOL_TIMEOUT = 1  # Timeout for ethtool subprocess calls
PROCESS_JOIN_TIMEOUT = 5  # Timeout for process join
PROCESS_TERMINATE_TIMEOUT = 1  # Timeout for process termination

# Plot styling constants
PLOT_DPI = 300  # DPI for saved plots
FONT_SIZE_DEFAULT = 14  # Default font size for plots
FONT_SIZE_TITLE = 20  # Font size for plot titles
FONT_SIZE_SUBTITLE = 18  # Font size for subplot titles
FONT_SIZE_AXIS_LABEL = 16  # Font size for axis labels
FONT_SIZE_LEGEND = 12  # Font size for legend


class MonitorType(Enum):
    """Enumeration of available monitoring types.

    Attributes:
        UTILIZATION: Monitor GPU utilization and power consumption.
        MEMORY: Monitor GPU memory usage.
        NETWORK: Monitor network bandwidth.
    """

    UTILIZATION = auto()
    MEMORY = auto()
    NETWORK = auto()


class CrystalMonitor:
    """Monitor for GPU utilization, memory usage, and network bandwidth.

    This class provides comprehensive monitoring capabilities for GPU metrics including:
    - GPU utilization and power consumption
    - GPU memory allocation tracking
    - Network bandwidth monitoring per GPU
    """

    def __init__(self, dir_name, worker_name, monitor_types=None, gpu_index=None, network_interfaces=None):
        """Initialize the CrystalMonitor instance.

        Args:
            dir_name: Directory path for saving monitoring data and plots.
            worker_name: Name identifier for this monitoring instance.
            monitor_types: List of MonitorType enums specifying which metrics to monitor.
                          If None, defaults to [MonitorType.MEMORY].
            gpu_index: Index of the GPU to monitor. Required when monitoring UTILIZATION or MEMORY.
            network_interfaces: List of network interface names to monitor. Required when monitoring NETWORK.
        """
        self.dir_name = dir_name
        Path(self.dir_name).mkdir(parents=True, exist_ok=True)
        self.worker_name = worker_name

        # Set default monitor types if not provided
        if monitor_types is None:
            monitor_types = [MonitorType.MEMORY]
        self.monitor_types = monitor_types

        # Validate gpu_index is provided when monitoring UTILIZATION or MEMORY
        if (
            MonitorType.UTILIZATION in self.monitor_types or MonitorType.MEMORY in self.monitor_types
        ) and gpu_index is None:
            raise ValueError("gpu_index must be explicitly specified when monitoring UTILIZATION or MEMORY")
        self.gpu_index = gpu_index

        # Validate network_interfaces is provided when monitoring NETWORK
        if MonitorType.NETWORK in self.monitor_types and network_interfaces is None:
            raise ValueError("network_interfaces must be explicitly specified when monitoring NETWORK")
        self.network_interfaces = network_interfaces

        # Initialize GPU capability info for dynamic Y-axis limits
        if self.gpu_index is not None:
            self._init_gpu_capabilities()
        else:
            self.power_y_max = 700  # Default power limit in Watts
            self.memory_y_max = 143  # Default memory in GB

        # Utilization monitoring process
        self.util_parent_conn = None
        self.util_child_conn = None
        self.util_process = None

        # Memory monitoring thread
        self.mem_thread = None
        self.mem_stop_event = None

        # Network monitoring processes
        self.net_parent_conn = None
        self.net_child_conn = None
        self.net_process = None

        # Build monitor type string for logging
        type_names = [t.name for t in self.monitor_types]
        gpu_info = f"GPU {self.gpu_index}" if self.gpu_index is not None else "No GPU"
        network_info = f"Network interfaces: {self.network_interfaces}. " if self.network_interfaces else ""
        print(
            f"Crystal Monitor {self.worker_name} initialized. "
            f"Monitor types: {type_names}. "
            f"GPU Info: {gpu_info}. "
            f"Network Info: {network_info}. "
            f"Plots will be saved to {self.dir_name}"
        )

    def _init_gpu_capabilities(self):
        """Initialize GPU capability info for dynamic Y-axis limits.

        Uses pynvml to query GPU power limit and total memory,
        which are used to set appropriate Y-axis maximum values for plots.
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

            # Get power limit (in milliwatts, convert to watts)
            power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            self.power_y_max = power_limit_mw / MILLIWATTS_TO_WATTS

            # Get total memory (in bytes, convert to GB)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.memory_y_max = mem_info.total / B_TO_GB

            pynvml.nvmlShutdown()
        except Exception:
            # Fallback to default values if pynvml fails
            self.power_y_max = 700  # Default power limit in Watts
            self.memory_y_max = 143  # Default memory in GB

    def start_measure(self):
        """Start monitoring for all configured monitor types."""
        if MonitorType.UTILIZATION in self.monitor_types:
            self._start_utilization_measure()
        if MonitorType.MEMORY in self.monitor_types:
            self._start_memory_measure()
        if MonitorType.NETWORK in self.monitor_types:
            self._start_network_measure()

    def end_measure(self):
        """End monitoring for all configured monitor types."""
        if MonitorType.UTILIZATION in self.monitor_types:
            self._end_utilization_measure()
        if MonitorType.MEMORY in self.monitor_types:
            self._end_memory_measure()
        if MonitorType.NETWORK in self.monitor_types:
            self._end_network_measure()

    def _utilization_worker(self, conn):
        try:
            import pynvml

            # Initialize metrics collection
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

            # Initialize data storage
            metrics_history = {"timestamps": deque(), "power": deque(), "gpu_util": deque()}

            interval = UTILIZATION_INTERVAL
            start_time = time.time()

            # Collect metrics until stop signal
            while True:
                try:
                    # Check for stop signal without blocking
                    if conn.poll(0):
                        msg = conn.recv()
                        if msg == "STOP":
                            break

                    current_time = time.time() - start_time

                    # Collect GPU metrics
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / MILLIWATTS_TO_WATTS  # W
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    metrics_history["timestamps"].append(current_time)
                    metrics_history["power"].append(power)
                    metrics_history["gpu_util"].append(gpu_util)

                    # Sleep for the remaining time to maintain interval
                    elapsed = time.time() - start_time - current_time
                    if elapsed < interval:
                        time.sleep(interval - elapsed)
                except Exception:
                    break

            # Create the GPU metrics plot
            timestamps = list(metrics_history["timestamps"])
            power = list(metrics_history["power"])
            gpu_util = list(metrics_history["gpu_util"])
            if timestamps:
                # Save raw GPU data
                util_data_file = self.dir_name + "/util_" + self.worker_name + ".txt"
                with Path(util_data_file).open("w", encoding="utf-8") as f:
                    f.write("Time(s)\tPower(W)\tGPU_Util(%)\n")
                    f.writelines(
                        f"{t:.2f}\t{p:.2f}\t{u:.2f}\n" for t, p, u in zip(timestamps, power, gpu_util, strict=True)
                    )

                plt.rcParams.update({"font.size": FONT_SIZE_DEFAULT})
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                fig.suptitle("GPU Metrics Over Time", fontsize=FONT_SIZE_TITLE, y=0.95)

                ax1.plot(timestamps, power, "r-", linewidth=2)
                ax1.set_xlabel("Time (s)", fontsize=FONT_SIZE_AXIS_LABEL)
                ax1.set_ylabel("Power (W)", fontsize=FONT_SIZE_AXIS_LABEL)
                ax1.set_title("GPU Power Usage", fontsize=FONT_SIZE_SUBTITLE, pad=10)
                # Set Y-axis limit to 1.1x of actual max power measured or GPU power limit
                power_y_max = max(max(power) * 1.1, self.power_y_max)
                ax1.set_ylim(0, power_y_max)
                ax1.grid(True)
                ax1.tick_params(axis="both", which="major", labelsize=FONT_SIZE_DEFAULT)

                ax2.plot(timestamps, gpu_util, "g-", linewidth=2)
                ax2.set_xlabel("Time (s)", fontsize=FONT_SIZE_AXIS_LABEL)
                ax2.set_ylabel("Utilization (%)", fontsize=FONT_SIZE_AXIS_LABEL)
                ax2.set_title("GPU Utilization", fontsize=FONT_SIZE_SUBTITLE, pad=10)
                ax2.set_ylim(0, UTILIZATION_Y_MAX)
                ax2.grid(True)
                ax2.tick_params(axis="both", which="major", labelsize=FONT_SIZE_DEFAULT)

                plt.tight_layout()
                plt.savefig(self.dir_name + "/util_" + self.worker_name + ".png", dpi=PLOT_DPI, bbox_inches="tight")
                plt.close()
            else:
                print("No metrics data collected")
        except Exception as e:
            print(f"Utilization Worker error: {e}")
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
            conn.close()

    def _start_utilization_measure(self):
        self.util_parent_conn, self.util_child_conn = Pipe()
        self.util_process = Process(target=self._utilization_worker, args=(self.util_child_conn,))
        self.util_process.start()

    def _end_utilization_measure(self):
        if self.util_process and self.util_process.is_alive():
            try:
                self.util_parent_conn.send("STOP")
                self.util_process.join(timeout=PROCESS_JOIN_TIMEOUT)
                if self.util_process.is_alive():
                    self.util_process.terminate()
                    self.util_process.join(timeout=PROCESS_TERMINATE_TIMEOUT)
            except Exception as e:
                print(f"Error during end_utilization_measure: {e}")
            finally:
                with contextlib.suppress(Exception):
                    self.util_parent_conn.close()

        print("Utilization monitoring completed.")

    def _memory_worker(self):
        try:
            import torch

            # Initialize data storage using metrics_history dict like other workers
            metrics_history = {"timestamps": deque(), "allocated": deque(), "max_allocated": deque()}

            interval = MEMORY_INTERVAL
            start_time = time.time()

            while not self.mem_stop_event.is_set():
                current_time = time.time() - start_time

                # Collect memory metrics using PyTorch
                allocated = torch.cuda.memory_allocated(self.gpu_index)
                max_allocated = torch.cuda.max_memory_allocated(self.gpu_index)

                metrics_history["timestamps"].append(current_time)
                metrics_history["allocated"].append(allocated / B_TO_GB)  # Convert to GB
                metrics_history["max_allocated"].append(max_allocated / B_TO_GB)  # Convert to GB

                # Sleep for the remaining time to maintain interval
                elapsed = time.time() - start_time - current_time
                if elapsed < interval:
                    time.sleep(interval - elapsed)

            # After stopping, save memory data
            timestamps = list(metrics_history["timestamps"])
            allocated = list(metrics_history["allocated"])
            max_allocated = list(metrics_history["max_allocated"])

            if timestamps:
                # Save raw memory data
                mem_data_file = self.dir_name + "/mem_" + self.worker_name + ".txt"
                with Path(mem_data_file).open("w", encoding="utf-8") as f:
                    f.write("Time(s)\tAllocated(GB)\tMax_Allocated(GB)\n")
                    f.writelines(
                        f"{t:.2f}\t{a:.2f}\t{m:.2f}\n"
                        for t, a, m in zip(timestamps, allocated, max_allocated, strict=True)
                    )

                plt.rcParams.update({"font.size": FONT_SIZE_DEFAULT})
                fig, ax = plt.subplots(figsize=(12, 6))
                fig.suptitle("GPU Memory Usage Over Time", fontsize=FONT_SIZE_TITLE, y=0.95)

                ax.plot(timestamps, allocated, "b-", linewidth=2, label="Allocated Memory")
                ax.plot(timestamps, max_allocated, "r--", linewidth=2, label="Max Allocated Memory")

                ax.set_xlabel("Time (s)", fontsize=FONT_SIZE_AXIS_LABEL)
                ax.set_ylabel("Memory (GB)", fontsize=FONT_SIZE_AXIS_LABEL)
                ax.set_title("GPU Memory Usage", fontsize=FONT_SIZE_SUBTITLE, pad=10)
                # Set Y-axis limit to 1.1x of actual max memory measured or GPU total memory
                memory_y_max = max(max(max_allocated) * 1.1, self.memory_y_max)
                ax.set_ylim(0, memory_y_max)
                ax.grid(True)
                ax.legend()
                ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_DEFAULT)

                plt.tight_layout()
                plt.savefig(self.dir_name + "/mem_" + self.worker_name + ".png", dpi=PLOT_DPI, bbox_inches="tight")
                plt.close()
        except Exception as e:
            print(f"Memory worker error: {e}")

    def _start_memory_measure(self):
        import torch

        torch.cuda.reset_max_memory_allocated(self.gpu_index)

        self.mem_stop_event = threading.Event()
        self.mem_thread = Thread(target=self._memory_worker)
        self.mem_thread.start()

    def _end_memory_measure(self):
        if self.mem_thread and self.mem_thread.is_alive():
            self.mem_stop_event.set()
            self.mem_thread.join()

        print("Memory monitoring completed.")

    def _get_all_network_bytes_ethtool(self):
        """Get tx_bytes_phy and rx_bytes_phy for all interfaces using a single ethtool call per interface.

        Uses subprocess.Popen for slightly better performance than subprocess.run.
        """
        results = {}

        # Start all ethtool processes in parallel
        processes = {}
        for interface in self.network_interfaces:
            try:
                proc = subprocess.Popen(
                    ["ethtool", "-S", interface],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                processes[interface] = proc
            except Exception:
                processes[interface] = None

        # Collect results from all processes
        for interface in self.network_interfaces:
            tx_bytes = 0
            rx_bytes = 0

            proc = processes.get(interface)
            if not proc:
                continue

            with contextlib.suppress(Exception):
                stdout, _ = proc.communicate(timeout=ETHOOL_TIMEOUT)
                for line in stdout.split("\n"):
                    if "tx_bytes_phy" in line:
                        match = re.search(r"tx_bytes_phy:\s*(\d+)", line)
                        if match:
                            tx_bytes = int(match.group(1))
                    elif "rx_bytes_phy" in line:
                        match = re.search(r"rx_bytes_phy:\s*(\d+)", line)
                        if match:
                            rx_bytes = int(match.group(1))

            results[interface] = {"tx": tx_bytes, "rx": rx_bytes}

        return results

    def _network_worker(self, conn):
        try:
            # Initialize metrics history for each interface
            metrics_history = {
                "timestamps": deque(),
            }
            for interface in self.network_interfaces:
                metrics_history[f"{interface}_tx"] = deque()
                metrics_history[f"{interface}_rx"] = deque()

            # Note: ethtool subprocess calls take ~30-50ms per interface, so actual
            # sampling interval will be limited by ethtool execution time.
            # Target interval is set to 1ms, but actual interval depends on system performance.
            interval = NETWORK_INTERVAL  # 1ms target sampling interval
            start_time = time.time()

            # Get initial bytes for all interfaces
            prev_bytes = self._get_all_network_bytes_ethtool()
            prev_time = start_time

            while True:
                try:
                    # Check for stop signal without blocking
                    if conn.poll(0):
                        msg = conn.recv()
                        if msg == "STOP":
                            break

                    current_time = time.time()
                    elapsed_since_start = current_time - start_time
                    elapsed_since_prev = current_time - prev_time

                    # Get current bytes for all interfaces
                    curr_bytes = self._get_all_network_bytes_ethtool()

                    # Record timestamp
                    metrics_history["timestamps"].append(elapsed_since_start)

                    # Calculate bandwidth for each interface
                    for interface in self.network_interfaces:
                        if elapsed_since_prev > 0:
                            tx_bandwidth = (
                                (curr_bytes[interface]["tx"] - prev_bytes[interface]["tx"])
                                * 8
                                / elapsed_since_prev
                                / BITS_TO_GIGABITS
                            )
                            rx_bandwidth = (
                                (curr_bytes[interface]["rx"] - prev_bytes[interface]["rx"])
                                * 8
                                / elapsed_since_prev
                                / BITS_TO_GIGABITS
                            )
                        else:
                            tx_bandwidth = 0
                            rx_bandwidth = 0

                        metrics_history[f"{interface}_tx"].append(tx_bandwidth)
                        metrics_history[f"{interface}_rx"].append(rx_bandwidth)

                    prev_bytes = curr_bytes
                    prev_time = current_time

                    # Sleep for remaining time to maintain 100us interval
                    sleep_time = interval - (time.time() - current_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                except Exception:
                    break

            # Save network metrics
            timestamps = list(metrics_history["timestamps"])

            if timestamps:
                # Save raw network data
                net_data_file = self.dir_name + "/network_" + self.worker_name + ".txt"
                with Path(net_data_file).open("w", encoding="utf-8") as f:
                    # Write header
                    header = "Time(s)"
                    for interface in self.network_interfaces:
                        header += f"\t{interface}_TX(Gbps)\t{interface}_RX(Gbps)"
                    header += "\tTotal_TX(Gbps)\tTotal_RX(Gbps)"
                    f.write(header + "\n")

                    # Write data
                    for i, t in enumerate(timestamps):
                        line = f"{t:.6f}"
                        tx_total = 0.0
                        rx_total = 0.0
                        for interface in self.network_interfaces:
                            tx = list(metrics_history[f"{interface}_tx"])[i]
                            rx = list(metrics_history[f"{interface}_rx"])[i]
                            line += f"\t{tx:.6f}\t{rx:.6f}"
                            tx_total += tx
                            rx_total += rx
                        line += f"\t{tx_total:.6f}\t{rx_total:.6f}"
                        f.write(line + "\n")

                    f.write(f"\nTotal execution time: {time.time() - start_time:.2f} seconds\n")
                    f.write(f"Total samples: {len(timestamps)}\n")
                    f.write(f"Monitored interfaces: {', '.join(self.network_interfaces)}\n")

                # Create network bandwidth plot with separate curves for each interface
                plt.rcParams.update({"font.size": FONT_SIZE_DEFAULT})
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                interfaces_str = ", ".join(self.network_interfaces)
                fig.suptitle(
                    f"Network Bandwidth Over Time (GPU {self.gpu_index}: {interfaces_str})",
                    fontsize=FONT_SIZE_SUBTITLE,
                    y=0.95,
                )

                # Color palette for different interfaces
                colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]

                # Calculate sum of all interfaces
                tx_sum = [0.0] * len(timestamps)
                rx_sum = [0.0] * len(timestamps)
                for interface in self.network_interfaces:
                    tx_data = list(metrics_history[f"{interface}_tx"])
                    rx_data = list(metrics_history[f"{interface}_rx"])
                    for i in range(len(timestamps)):
                        tx_sum[i] += tx_data[i]
                        rx_sum[i] += rx_data[i]

                # TX bandwidth plot
                max_tx = 0
                for idx, interface in enumerate(self.network_interfaces):
                    tx_data = list(metrics_history[f"{interface}_tx"])
                    ax1.plot(
                        timestamps,
                        tx_data,
                        color=colors[idx % len(colors)],
                        linewidth=1,
                        alpha=0.7,
                        label=f"{interface}",
                    )
                    if tx_data:
                        max_tx = max(max_tx, *tx_data)

                # Add sum line for TX
                ax1.plot(timestamps, tx_sum, color="black", linewidth=2, alpha=0.9, linestyle="--", label="Total")
                max_tx = max(max_tx, max(tx_sum) if tx_sum else 0)

                ax1.set_ylabel("TX Bandwidth (Gbps)", fontsize=FONT_SIZE_AXIS_LABEL)
                ax1.set_title("TX Bandwidth (tx_bytes)", fontsize=FONT_SIZE_AXIS_LABEL, pad=10)
                ax1.set_ylim(0, max(max_tx * 1.1, 1))
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
                ax1.tick_params(axis="both", which="major", labelsize=FONT_SIZE_LEGEND)

                # RX bandwidth plot
                max_rx = 0
                for idx, interface in enumerate(self.network_interfaces):
                    rx_data = list(metrics_history[f"{interface}_rx"])
                    ax2.plot(
                        timestamps,
                        rx_data,
                        color=colors[idx % len(colors)],
                        linewidth=1,
                        alpha=0.7,
                        label=f"{interface}",
                    )
                    if rx_data:
                        max_rx = max(max_rx, *rx_data)

                # Add sum line for RX
                ax2.plot(timestamps, rx_sum, color="black", linewidth=2, alpha=0.9, linestyle="--", label="Total")
                max_rx = max(max_rx, max(rx_sum) if rx_sum else 0)

                ax2.set_xlabel("Time (s)", fontsize=FONT_SIZE_AXIS_LABEL)
                ax2.set_ylabel("RX Bandwidth (Gbps)", fontsize=FONT_SIZE_AXIS_LABEL)
                ax2.set_title("RX Bandwidth (rx_bytes)", fontsize=FONT_SIZE_AXIS_LABEL, pad=10)
                ax2.set_ylim(0, max(max_rx * 1.1, 1))
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
                ax2.tick_params(axis="both", which="major", labelsize=FONT_SIZE_LEGEND)

                plt.tight_layout()
                plt.savefig(self.dir_name + "/network_" + self.worker_name + ".png", dpi=PLOT_DPI, bbox_inches="tight")
                plt.close()

                print(f"Network monitoring saved {len(timestamps)} samples for interfaces: {interfaces_str}")
            else:
                print("No network metrics data collected")
        except Exception as e:
            print(f"Network worker error: {e}")
        finally:
            conn.close()

    def _start_network_measure(self):
        self.net_parent_conn, self.net_child_conn = Pipe()
        self.net_process = Process(target=self._network_worker, args=(self.net_child_conn,))
        self.net_process.start()
        print(f"Network monitoring started for interfaces: {', '.join(self.network_interfaces)}")

    def _end_network_measure(self):
        if self.net_process and self.net_process.is_alive():
            try:
                self.net_parent_conn.send("STOP")
                self.net_process.join(timeout=PROCESS_JOIN_TIMEOUT)
                if self.net_process.is_alive():
                    self.net_process.terminate()
                    self.net_process.join(timeout=PROCESS_TERMINATE_TIMEOUT)
            except Exception as e:
                print(f"Error during end_network_measure: {e}")
            finally:
                with contextlib.suppress(Exception):
                    self.net_parent_conn.close()
        print("Network monitoring completed")
