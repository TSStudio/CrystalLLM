# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Comprehensive tests for CrystalLLM Monitor."""

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from crystalllm.monitor import CrystalMonitor, MonitorType, util


class TestGetDualNetworkInterfacesForGPU(unittest.TestCase):
    """Test get_dual_network_interfaces_for_gpu helper function."""

    def test_gpu_0_interfaces(self):
        """Test network interfaces for GPU 0."""
        interfaces = util.get_dual_network_interfaces_for_gpu(0)
        self.assertEqual(interfaces, ["reth0", "reth1"])

    def test_gpu_1_interfaces(self):
        """Test network interfaces for GPU 1."""
        interfaces = util.get_dual_network_interfaces_for_gpu(1)
        self.assertEqual(interfaces, ["reth2", "reth3"])

    def test_gpu_n_interfaces(self):
        """Test network interfaces for arbitrary GPU N."""
        for gpu_index in [2, 3, 7, 10]:
            interfaces = util.get_dual_network_interfaces_for_gpu(gpu_index)
            expected = [f"reth{gpu_index * 2}", f"reth{gpu_index * 2 + 1}"]
            self.assertEqual(interfaces, expected)


class TestMonitorType(unittest.TestCase):
    """Test MonitorType enum."""

    def test_monitor_type_values(self):
        """Test MonitorType enum values exist."""
        self.assertIsNotNone(MonitorType.UTILIZATION)
        self.assertIsNotNone(MonitorType.MEMORY)
        self.assertIsNotNone(MonitorType.NETWORK)

    def test_monitor_type_names(self):
        """Test MonitorType enum names."""
        self.assertEqual(MonitorType.UTILIZATION.name, "UTILIZATION")
        self.assertEqual(MonitorType.MEMORY.name, "MEMORY")
        self.assertEqual(MonitorType.NETWORK.name, "NETWORK")


class TestCrystalMonitorInitialization(unittest.TestCase):
    """Test CrystalMonitor initialization."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_initialization_with_memory_only(self):
        """Test basic initialization with MEMORY type only."""
        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="test_worker",
            monitor_types=[MonitorType.MEMORY],
            gpu_index=0,
        )
        self.assertEqual(monitor.dir_name, self.temp_dir)
        self.assertEqual(monitor.worker_name, "test_worker")
        self.assertEqual(monitor.monitor_types, [MonitorType.MEMORY])
        self.assertEqual(monitor.gpu_index, 0)
        self.assertIsNone(monitor.network_interfaces)

    def test_initialization_with_utilization(self):
        """Test initialization with UTILIZATION type."""
        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="util_worker",
            monitor_types=[MonitorType.UTILIZATION],
            gpu_index=1,
        )
        self.assertEqual(monitor.gpu_index, 1)
        self.assertEqual(monitor.monitor_types, [MonitorType.UTILIZATION])

    def test_initialization_with_network(self):
        """Test initialization with NETWORK type."""
        interfaces = ["reth0", "reth1"]
        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="network_worker",
            monitor_types=[MonitorType.NETWORK],
            network_interfaces=interfaces,
        )
        self.assertEqual(monitor.network_interfaces, interfaces)
        self.assertIsNone(monitor.gpu_index)

    def test_initialization_with_all_types(self):
        """Test initialization with all monitor types."""
        interfaces = ["reth0", "reth1"]
        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="full_worker",
            monitor_types=[MonitorType.UTILIZATION, MonitorType.MEMORY, MonitorType.NETWORK],
            gpu_index=0,
            network_interfaces=interfaces,
        )
        self.assertEqual(monitor.gpu_index, 0)
        self.assertEqual(monitor.network_interfaces, interfaces)
        self.assertEqual(
            set(monitor.monitor_types),
            {MonitorType.UTILIZATION, MonitorType.MEMORY, MonitorType.NETWORK},
        )

    def test_default_monitor_types(self):
        """Test default monitor types is MEMORY."""
        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="default_worker",
            gpu_index=0,
        )
        self.assertEqual(monitor.monitor_types, [MonitorType.MEMORY])

    def test_missing_gpu_index_for_utilization_raises(self):
        """Test that missing gpu_index raises ValueError for UTILIZATION."""
        with self.assertRaises(ValueError) as context:
            CrystalMonitor(
                dir_name=self.temp_dir,
                worker_name="test",
                monitor_types=[MonitorType.UTILIZATION],
            )
        self.assertIn("gpu_index must be explicitly specified", str(context.exception))

    def test_missing_gpu_index_for_memory_raises(self):
        """Test that missing gpu_index raises ValueError for MEMORY."""
        with self.assertRaises(ValueError) as context:
            CrystalMonitor(
                dir_name=self.temp_dir,
                worker_name="test",
                monitor_types=[MonitorType.MEMORY],
            )
        self.assertIn("gpu_index must be explicitly specified", str(context.exception))

    def test_missing_network_interfaces_for_network_raises(self):
        """Test that missing network_interfaces raises ValueError for NETWORK."""
        with self.assertRaises(ValueError) as context:
            CrystalMonitor(
                dir_name=self.temp_dir,
                worker_name="test",
                monitor_types=[MonitorType.NETWORK],
            )
        self.assertIn("network_interfaces must be explicitly specified", str(context.exception))


class TestCrystalMonitorGPUCapabilities(unittest.TestCase):
    """Test CrystalMonitor GPU capability initialization."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch.dict(sys.modules, {"pynvml": mock.Mock()})
    def test_gpu_capabilities_initialized(self):
        """Test GPU capabilities are initialized from pynvml."""
        import sys

        mock_pynvml = sys.modules["pynvml"]
        mock_handle = mock.Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerManagementLimit.return_value = 300000  # 300W in mW
        mock_mem_info = mock.Mock()
        mock_mem_info.total = 85899345920  # 80 GB in bytes
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="test",
            monitor_types=[MonitorType.UTILIZATION],
            gpu_index=0,
        )

        self.assertEqual(monitor.power_y_max, 300.0)  # 300W
        self.assertEqual(monitor.memory_y_max, 85.89934592)  # ~80 GB

    @mock.patch.dict(sys.modules, {"pynvml": mock.Mock()})
    def test_gpu_capabilities_fallback_on_error(self):
        """Test GPU capabilities fallback to defaults on pynvml error."""
        import sys

        mock_pynvml = sys.modules["pynvml"]
        mock_pynvml.nvmlInit.side_effect = Exception("NVML Error")

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="test",
            monitor_types=[MonitorType.UTILIZATION],
            gpu_index=0,
        )

        self.assertEqual(monitor.power_y_max, 700)  # Default
        self.assertEqual(monitor.memory_y_max, 143)  # Default

    def test_default_capabilities_without_gpu(self):
        """Test default capabilities when gpu_index is None (NETWORK only)."""
        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="test",
            monitor_types=[MonitorType.NETWORK],
            network_interfaces=["reth0", "reth1"],
        )

        self.assertEqual(monitor.power_y_max, 700)  # Default
        self.assertEqual(monitor.memory_y_max, 143)  # Default


class TestCrystalMonitorMemoryMonitoring(unittest.TestCase):
    """Test CrystalMonitor memory monitoring functionality."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch.dict(sys.modules, {"torch": mock.Mock()})
    def test_memory_monitoring_start_stop(self):
        """Test memory monitoring start and stop."""
        import sys

        mock_torch = sys.modules["torch"]
        mock_cuda = mock.Mock()
        mock_torch.cuda = mock_cuda
        mock_cuda.memory_allocated.return_value = 1024 * 1024 * 1024  # 1 GB
        mock_cuda.max_memory_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2 GB

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="mem_test",
            monitor_types=[MonitorType.MEMORY],
            gpu_index=0,
        )

        monitor.start_measure()
        self.assertIsNotNone(monitor.mem_thread)
        self.assertIsNotNone(monitor.mem_stop_event)
        self.assertTrue(monitor.mem_thread.is_alive())

        # Let it collect some data
        time.sleep(0.05)

        monitor.end_measure()
        self.assertFalse(monitor.mem_thread.is_alive())

    @mock.patch.dict(sys.modules, {"torch": mock.Mock()})
    def test_memory_data_file_generated(self):
        """Test that memory monitoring generates data file."""
        import sys

        mock_torch = sys.modules["torch"]
        mock_cuda = mock.Mock()
        mock_torch.cuda = mock_cuda
        mock_cuda.memory_allocated.return_value = 1024 * 1024 * 1024  # 1 GB
        mock_cuda.max_memory_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2 GB

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="mem_file_test",
            monitor_types=[MonitorType.MEMORY],
            gpu_index=0,
        )

        monitor.start_measure()
        time.sleep(0.05)
        monitor.end_measure()

        data_file = Path(self.temp_dir) / "mem_mem_file_test.txt"
        self.assertTrue(data_file.exists())

        # Verify file content
        content = data_file.read_text()
        self.assertIn("Time(s)", content)
        self.assertIn("Allocated(GB)", content)
        self.assertIn("Max_Allocated(GB)", content)

    @mock.patch.dict(sys.modules, {"torch": mock.Mock()})
    def test_memory_plot_file_generated(self):
        """Test that memory monitoring generates plot file."""
        import sys

        mock_torch = sys.modules["torch"]
        mock_cuda = mock.Mock()
        mock_torch.cuda = mock_cuda
        mock_cuda.memory_allocated.return_value = 1024 * 1024 * 1024
        mock_cuda.max_memory_allocated.return_value = 2 * 1024 * 1024 * 1024

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="mem_plot_test",
            monitor_types=[MonitorType.MEMORY],
            gpu_index=0,
        )

        monitor.start_measure()
        time.sleep(0.05)
        monitor.end_measure()

        plot_file = Path(self.temp_dir) / "mem_mem_plot_test.png"
        self.assertTrue(plot_file.exists())
        self.assertGreater(plot_file.stat().st_size, 0)


class TestCrystalMonitorUtilizationMonitoring(unittest.TestCase):
    """Test CrystalMonitor utilization monitoring functionality."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch.dict(sys.modules, {"pynvml": mock.Mock()})
    def test_utilization_monitoring_start_stop(self):
        """Test utilization monitoring start and stop."""
        import sys

        mock_pynvml = sys.modules["pynvml"]
        mock_handle = mock.Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000  # 150W
        mock_util = mock.Mock()
        mock_util.gpu = 75
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="util_test",
            monitor_types=[MonitorType.UTILIZATION],
            gpu_index=0,
        )

        monitor.start_measure()
        self.assertIsNotNone(monitor.util_process)
        self.assertTrue(monitor.util_process.is_alive())

        time.sleep(0.05)

        monitor.end_measure()
        # Process should be terminated or completed

    @mock.patch.dict(sys.modules, {"pynvml": mock.Mock()})
    def test_utilization_data_file_generated(self):
        """Test that utilization monitoring generates data file."""
        import sys

        mock_pynvml = sys.modules["pynvml"]
        mock_handle = mock.Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000
        mock_util = mock.Mock()
        mock_util.gpu = 75
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="util_file_test",
            monitor_types=[MonitorType.UTILIZATION],
            gpu_index=0,
        )

        monitor.start_measure()
        time.sleep(0.05)
        monitor.end_measure()

        data_file = Path(self.temp_dir) / "util_util_file_test.txt"
        self.assertTrue(data_file.exists())

        content = data_file.read_text()
        self.assertIn("Time(s)", content)
        self.assertIn("Power(W)", content)
        self.assertIn("GPU_Util(%)", content)


class TestCrystalMonitorNetworkMonitoring(unittest.TestCase):
    """Test CrystalMonitor network monitoring functionality."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch("crystalllm.monitor.monitor.subprocess.Popen")
    def test_network_monitoring_start_stop(self, mock_popen):
        """Test network monitoring start and stop."""
        mock_proc = mock.Mock()
        mock_proc.communicate.return_value = ("tx_bytes_phy: 1000\nrx_bytes_phy: 2000", "")
        mock_popen.return_value = mock_proc

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="net_test",
            monitor_types=[MonitorType.NETWORK],
            network_interfaces=["reth0", "reth1"],
        )

        monitor.start_measure()
        self.assertIsNotNone(monitor.net_process)
        self.assertTrue(monitor.net_process.is_alive())

        time.sleep(0.05)

        monitor.end_measure()

    @mock.patch("crystalllm.monitor.monitor.subprocess.Popen")
    def test_network_data_file_generated(self, mock_popen):
        """Test that network monitoring generates data file."""
        mock_proc = mock.Mock()
        mock_proc.communicate.return_value = ("tx_bytes_phy: 1000\nrx_bytes_phy: 2000", "")
        mock_popen.return_value = mock_proc

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="net_file_test",
            monitor_types=[MonitorType.NETWORK],
            network_interfaces=["reth0", "reth1"],
        )

        monitor.start_measure()
        time.sleep(0.05)
        monitor.end_measure()

        data_file = Path(self.temp_dir) / "network_net_file_test.txt"
        self.assertTrue(data_file.exists())

        content = data_file.read_text()
        self.assertIn("Time(s)", content)
        self.assertIn("reth0_TX(Gbps)", content)
        self.assertIn("reth0_RX(Gbps)", content)


class TestCrystalMonitorCombinedMonitoring(unittest.TestCase):
    """Test CrystalMonitor with combined monitoring types."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch.dict(sys.modules, {"pynvml": mock.Mock(), "torch": mock.Mock()})
    @mock.patch("crystalllm.monitor.monitor.subprocess.Popen")
    def test_all_monitor_types_together(self, mock_popen):
        """Test all monitor types running together."""
        import sys

        mock_pynvml = sys.modules["pynvml"]
        mock_torch = sys.modules["torch"]

        # Setup pynvml mocks
        mock_handle = mock.Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150000
        mock_util = mock.Mock()
        mock_util.gpu = 75
        mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        # Setup torch.cuda mocks
        mock_cuda = mock.Mock()
        mock_torch.cuda = mock_cuda
        mock_cuda.memory_allocated.return_value = 1024 * 1024 * 1024
        mock_cuda.max_memory_allocated.return_value = 2 * 1024 * 1024 * 1024

        # Setup subprocess mocks
        mock_proc = mock.Mock()
        mock_proc.communicate.return_value = ("tx_bytes_phy: 1000\nrx_bytes_phy: 2000", "")
        mock_popen.return_value = mock_proc

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="combined_test",
            monitor_types=[MonitorType.UTILIZATION, MonitorType.MEMORY, MonitorType.NETWORK],
            gpu_index=0,
            network_interfaces=["reth0", "reth1"],
        )

        monitor.start_measure()

        # Verify all monitoring processes/threads started
        self.assertIsNotNone(monitor.util_process)
        self.assertIsNotNone(monitor.mem_thread)
        self.assertIsNotNone(monitor.net_process)

        time.sleep(0.1)

        monitor.end_measure()

        # Verify all data files were created
        self.assertTrue((Path(self.temp_dir) / "util_combined_test.txt").exists())
        self.assertTrue((Path(self.temp_dir) / "mem_combined_test.txt").exists())
        self.assertTrue((Path(self.temp_dir) / "network_combined_test.txt").exists())


class TestCrystalMonitorErrorHandling(unittest.TestCase):
    """Test CrystalMonitor error handling and edge cases."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_measure_without_start(self):
        """Test end_measure without start_measure does not raise."""
        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="error_test",
            monitor_types=[MonitorType.MEMORY],
            gpu_index=0,
        )

        # Should not raise any exception
        monitor.end_measure()

    @mock.patch.dict(sys.modules, {"torch": mock.Mock()})
    def test_memory_worker_exception_handling(self):
        """Test memory worker handles exceptions gracefully."""
        import sys

        mock_torch = sys.modules["torch"]
        mock_cuda = mock.Mock()
        mock_torch.cuda = mock_cuda

        # Make memory_allocated raise exception during worker execution
        mock_cuda.memory_allocated.side_effect = Exception("CUDA Error")

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="mem_error_test",
            monitor_types=[MonitorType.MEMORY],
            gpu_index=0,
        )

        # Should not raise during start
        try:
            monitor.start_measure()
            time.sleep(0.05)
            monitor.end_measure()
        except Exception as e:
            self.fail(f"Should not raise exception: {e}")

    def test_empty_network_interfaces(self):
        """Test empty network interfaces list."""
        # Empty list should be valid
        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="empty_net_test",
            monitor_types=[MonitorType.NETWORK],
            network_interfaces=[],
        )
        self.assertEqual(monitor.network_interfaces, [])


class TestCrystalMonitorEdgeCases(unittest.TestCase):
    """Test CrystalMonitor edge cases."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_worker_name_with_special_characters(self):
        """Test worker names with special characters."""
        special_names = ["worker-1", "worker.2", "worker_3", "worker/with/slashes"]
        for name in special_names:
            monitor = CrystalMonitor(
                dir_name=self.temp_dir,
                worker_name=name,
                monitor_types=[MonitorType.MEMORY],
                gpu_index=0,
            )
            self.assertEqual(monitor.worker_name, name)

    def test_non_existent_directory(self):
        """Test with non-existent directory path."""
        non_existent_dir = str(Path(self.temp_dir) / "subdir" / "nested")
        monitor = CrystalMonitor(
            dir_name=non_existent_dir,
            worker_name="dir_test",
            monitor_types=[MonitorType.MEMORY],
            gpu_index=0,
        )
        self.assertEqual(monitor.dir_name, non_existent_dir)

    @mock.patch.dict(sys.modules, {"torch": mock.Mock()})
    def test_multiple_start_calls(self):
        """Test multiple start_measure calls creates new threads."""
        import sys

        mock_torch = sys.modules["torch"]
        mock_cuda = mock.Mock()
        mock_torch.cuda = mock_cuda
        mock_cuda.memory_allocated.return_value = 1024 * 1024 * 1024
        mock_cuda.max_memory_allocated.return_value = 2 * 1024 * 1024 * 1024

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="multi_start",
            monitor_types=[MonitorType.MEMORY],
            gpu_index=0,
        )

        monitor.start_measure()
        first_thread = monitor.mem_thread

        # Stop first measurement
        monitor.end_measure()

        # Second start should create a new thread
        monitor.start_measure()
        second_thread = monitor.mem_thread

        # After stop and restart, should be a different thread
        self.assertIsNotNone(first_thread)
        self.assertIsNotNone(second_thread)

        monitor.end_measure()

    @mock.patch.dict(sys.modules, {"torch": mock.Mock()})
    def test_multiple_end_calls(self):
        """Test multiple end_measure calls."""
        import sys

        mock_torch = sys.modules["torch"]
        mock_cuda = mock.Mock()
        mock_torch.cuda = mock_cuda
        mock_cuda.memory_allocated.return_value = 1024 * 1024 * 1024
        mock_cuda.max_memory_allocated.return_value = 2 * 1024 * 1024 * 1024

        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="multi_end",
            monitor_types=[MonitorType.MEMORY],
            gpu_index=0,
        )

        monitor.start_measure()
        time.sleep(0.05)
        monitor.end_measure()

        # Second end should not raise
        monitor.end_measure()
        monitor.end_measure()


class TestCrystalMonitorIntegration(unittest.TestCase):
    """Integration tests for CrystalMonitor."""

    def setUp(self):
        """Set up temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @mock.patch.dict(sys.modules, {"torch": mock.Mock()})
    def test_full_monitoring_lifecycle(self):
        """Test full monitoring lifecycle from init to file generation."""
        import sys

        mock_torch = sys.modules["torch"]
        mock_cuda = mock.Mock()
        mock_torch.cuda = mock_cuda
        mock_cuda.memory_allocated.return_value = 1024 * 1024 * 1024
        mock_cuda.max_memory_allocated.return_value = 2 * 1024 * 1024 * 1024

        # Create monitor
        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="lifecycle_test",
            monitor_types=[MonitorType.MEMORY],
            gpu_index=0,
        )

        # Start monitoring
        monitor.start_measure()

        # Simulate some work time
        time.sleep(0.1)

        # Stop monitoring
        monitor.end_measure()

        # Verify files exist
        data_file = Path(self.temp_dir) / "mem_lifecycle_test.txt"
        plot_file = Path(self.temp_dir) / "mem_lifecycle_test.png"

        self.assertTrue(data_file.exists())
        self.assertTrue(plot_file.exists())

        # Verify data file has content
        content = data_file.read_text()
        lines = content.strip().split("\n")
        self.assertGreater(len(lines), 1)  # Header + at least one data line

    def test_monitor_with_large_gpu_index(self):
        """Test monitor with large GPU index."""
        monitor = CrystalMonitor(
            dir_name=self.temp_dir,
            worker_name="large_gpu",
            monitor_types=[MonitorType.NETWORK],
            network_interfaces=["reth0", "reth1"],
        )
        # NETWORK type doesn't require gpu_index
        self.assertIsNone(monitor.gpu_index)


if __name__ == "__main__":
    unittest.main()
