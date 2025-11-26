import sys
from unittest.mock import MagicMock

# torch._dynamo mock
import torch
if not hasattr(torch, '_dynamo'):
    torch._dynamo = MagicMock()
    torch._dynamo.disable = lambda: lambda fn: fn

# kaolin mock (전체 하위 모듈)
kaolin_mock = MagicMock()
sys.modules["kaolin"] = kaolin_mock
sys.modules["kaolin.visualize"] = kaolin_mock
sys.modules["kaolin.render"] = kaolin_mock
sys.modules["kaolin.render.camera"] = kaolin_mock
sys.modules["kaolin.physics"] = kaolin_mock
sys.modules["kaolin.utils"] = kaolin_mock
sys.modules["kaolin.utils.testing"] = kaolin_mock

# lightning mock (전체 하위 모듈)
# isinstance() 체크를 위해 실제 클래스 타입 필요
class MockLightningModule:
    """Mock class for pl.LightningModule isinstance() checks"""
    pass

lightning_mock = MagicMock()
lightning_mock.LightningModule = MockLightningModule

lightning_pytorch_mock = MagicMock()
lightning_pytorch_mock.LightningModule = MockLightningModule

sys.modules["lightning"] = lightning_mock
sys.modules["lightning.pytorch"] = lightning_pytorch_mock
sys.modules["lightning.pytorch.utilities"] = MagicMock()
sys.modules["lightning.pytorch.utilities.consolidate_checkpoint"] = MagicMock()

print("patch applied")
