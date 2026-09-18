import dataclasses

from securesd.config import Config
from securesd.engine.model_runner import ModelRunner


class DraftRunner(ModelRunner):
    def __init__(self, config: Config):
        self.draft_config = dataclasses.replace(
            config,
            model=config.draft,
            gpu_memory_utilization=config.draft_gpu_memory_utilization,
        )
        super().__init__(self.draft_config, is_draft=True)
