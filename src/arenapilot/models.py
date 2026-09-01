from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class MetricDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class SplitType(StrEnum):
    KFOLD = "kfold"
    STRATIFIED_KFOLD = "stratified_kfold"
    GROUP_KFOLD = "group_kfold"
    STRATIFIED_GROUP_KFOLD = "stratified_group_kfold"
    TIME_SERIES = "time_series"


class PredictionType(StrEnum):
    PROBABILITY = "probability"
    LABEL = "label"
    VALUE = "value"


class CompetitionConfig(StrictModel):
    platform: Literal["kaggle"]
    slug: str = Field(min_length=1)
    title: str | None = None
    status: Literal["draft", "ready"] = "draft"


class TaskConfig(StrictModel):
    type: TaskType
    target: str = Field(min_length=1)


class MetricConfig(StrictModel):
    name: str = Field(min_length=1)
    direction: MetricDirection


class ValidationRef(StrictModel):
    active: str | None = None


class KaggleComputeConfig(StrictModel):
    enabled: bool = True
    accelerator: str | None = "gpu"
    internet: bool = False


class ComputeConfig(StrictModel):
    default_backend: Literal["local", "kaggle"] = "local"
    kaggle: KaggleComputeConfig = Field(default_factory=KaggleComputeConfig)


class TrackingConfig(StrictModel):
    backend: Literal["mlflow"] = "mlflow"
    experiment_name: str = Field(min_length=1)


class SubmissionConfig(StrictModel):
    mode: Literal["manual", "budgeted_auto"] = "manual"
    daily_budget: int = Field(default=3, ge=0)
    total_budget: int = Field(default=30, ge=0)
    require_verified_run: bool = True

    @model_validator(mode="after")
    def total_budget_covers_daily_budget(self) -> "SubmissionConfig":
        if self.total_budget and self.daily_budget > self.total_budget:
            raise ValueError("daily_budget cannot exceed total_budget")
        return self


class MemoryConfig(StrictModel):
    enabled: bool = True
    retrieve_on_init: bool = True


class WorkspaceConfig(StrictModel):
    default_seed: int = 42


class ArenaConfig(StrictModel):
    schema_version: Literal[1] = 1
    competition: CompetitionConfig
    task: TaskConfig | None = None
    metric: MetricConfig | None = None
    validation: ValidationRef = Field(default_factory=ValidationRef)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    tracking: TrackingConfig
    submission: SubmissionConfig = Field(default_factory=SubmissionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)

    @model_validator(mode="after")
    def ready_competition_requires_evaluation_contract(self) -> "ArenaConfig":
        if self.competition.status == "ready":
            missing: list[str] = []
            if self.task is None:
                missing.append("task")
            if self.metric is None:
                missing.append("metric")
            if self.validation.active is None:
                missing.append("validation.active")
            if missing:
                raise ValueError(
                    "ready competition requires configured " + ", ".join(missing)
                )
        return self


class SplitConfig(StrictModel):
    type: SplitType
    n_splits: int = Field(default=5, ge=2)
    shuffle: bool = True
    random_state: int | None = 42
    group_column: str | None = None
    time_column: str | None = None

    @model_validator(mode="after")
    def validate_split_requirements(self) -> "SplitConfig":
        if self.type in {SplitType.GROUP_KFOLD, SplitType.STRATIFIED_GROUP_KFOLD} and not self.group_column:
            raise ValueError("group_column is required for group-based splits")
        if self.type == SplitType.TIME_SERIES and not self.time_column:
            raise ValueError("time_column is required for time_series splits")
        return self


class PredictionConfig(StrictModel):
    type: PredictionType


class OOFConfig(StrictModel):
    required: bool = True
    require_exactly_once: bool = True


class ValidationSpec(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^val-v[1-9][0-9]*$")
    parent: str | None = None
    status: Literal["draft", "active", "deprecated"] = "draft"
    reason: str = Field(min_length=1)
    split: SplitConfig
    metric: MetricConfig | None = None
    prediction: PredictionConfig | None = None
    oof: OOFConfig = Field(default_factory=OOFConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def active_validation_requires_metric_and_prediction(self) -> "ValidationSpec":
        if self.status == "active":
            missing: list[str] = []
            if self.metric is None:
                missing.append("metric")
            if self.prediction is None:
                missing.append("prediction")
            if missing:
                raise ValueError(
                    "active validation requires configured " + ", ".join(missing)
                )
        return self


class ExperimentParent(StrictModel):
    experiment: str = Field(pattern=r"^exp[0-9]{3,}$")
    relation: Literal["derived_from", "ablation_of", "reproduction_of", "ensemble_of"] = "derived_from"


class SeedConfig(StrictModel):
    policy: Literal["fixed"] = "fixed"
    value: int = 42


class ExperimentRuntime(StrictModel):
    backend: Literal["local", "kaggle"] = "local"
    resources: dict[str, Any] = Field(default_factory=dict)


class ExperimentSpec(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^exp[0-9]{3,}$")
    title: str = Field(min_length=1)
    parents: list[ExperimentParent] = Field(default_factory=list)
    hypothesis: str = Field(min_length=1)
    validation: str = Field(pattern=r"^val-v[1-9][0-9]*$")
    pipeline: dict[str, Any] = Field(default_factory=dict)
    model: dict[str, Any]
    seed: SeedConfig = Field(default_factory=SeedConfig)
    runtime: ExperimentRuntime = Field(default_factory=ExperimentRuntime)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
