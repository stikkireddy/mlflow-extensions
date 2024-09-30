from dataclasses import dataclass, field

from mlflow_extensions.databricks.prebuilt.embeddings.infinity import InfinityEmbeddings


@dataclass(frozen=True)
class EmbeddingModels:
    infinity: InfinityEmbeddings = field(default_factory=lambda: InfinityEmbeddings())
