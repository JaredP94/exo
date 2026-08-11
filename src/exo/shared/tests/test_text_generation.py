from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams


def test_prefix_cache_is_enabled_by_default_for_normal_tasks() -> None:
    task = TextGenerationTaskParams(model=ModelId("test"), input=[])

    assert task.use_prefix_cache is True
