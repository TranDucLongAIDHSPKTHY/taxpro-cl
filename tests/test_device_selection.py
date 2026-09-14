import unittest
from unittest.mock import Mock

import torch

from main import resolve_device


class DeviceSelectionTests(unittest.TestCase):
    def fake_torch(self, available=False, count=0):
        module = Mock()
        module.device.side_effect = torch.device
        module.cuda.is_available.return_value = available
        module.cuda.device_count.return_value = count
        return module

    def test_auto_falls_back_to_cpu(self):
        device = resolve_device("auto", torch_module=self.fake_torch())
        self.assertEqual(device, torch.device("cpu"))

    def test_cpu_is_honored_when_cuda_exists(self):
        device = resolve_device(
            "cpu", torch_module=self.fake_torch(available=True, count=1)
        )
        self.assertEqual(device, torch.device("cpu"))

    def test_explicit_cuda_fails_when_unavailable(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            resolve_device("cuda", torch_module=self.fake_torch())

    def test_cuda_uses_selected_gpu(self):
        device = resolve_device(
            "cuda", gpu_id=1, torch_module=self.fake_torch(available=True, count=2)
        )
        self.assertEqual(device, torch.device("cuda:1"))

    def test_legacy_cuda_false_selects_cpu(self):
        device = resolve_device(
            "auto",
            legacy_cuda=False,
            torch_module=self.fake_torch(available=True, count=1),
        )
        self.assertEqual(device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
