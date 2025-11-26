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
lightning_mock = MagicMock()
sys.modules["lightning"] = lightning_mock
sys.modules["lightning.pytorch"] = lightning_mock
sys.modules["lightning.pytorch.utilities"] = lightning_mock
sys.modules["lightning.pytorch.utilities.consolidate_checkpoint"] = lightning_mock

print("patch applied")
