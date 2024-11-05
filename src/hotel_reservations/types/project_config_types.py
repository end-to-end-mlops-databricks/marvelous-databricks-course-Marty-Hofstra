from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Constraints(BaseModel):
    min: Optional[Union[int, float]]  # Using Union for both int and float types
    max: Optional[Union[int, float]]  # Optional in case it is not defined


class NumFeature(BaseModel):
    type: str
    constraints: Constraints


class CatFeature(BaseModel):
    type: str
    allowed_values: List[Union[str, bool]]  # Can include strings or booleans
    encoding: Optional[List[int]]  # Optional encoding


class Parameters(BaseModel):
    """
    Holds model parameters.
    - `learning_rate`: Learning rate for the model (e.g., 0.01).
    - `n_estimators`: Number of estimators (e.g., 1000).
    - `max_depth`: Maximum tree depth (e.g., 6).
    """

    learning_rate: float = Field(..., gt=0, le=1)  # Use Field to set constraints
    n_estimators: int = Field(..., gt=0, le=10000)  # Use Field to set constraints
    max_depth: int = Field(..., gt=0, le=32)  # Use Field to set constraints


class ProjectConfig(BaseModel):
    """
    Defines the configuration for the project.
    - `catalog`: The data catalog name.
    - `db_schema`: The schema where the dataset resides. Alias to `schema`
    - `table_name`: The table name of the dataset.
    - `parameters`: Model parameters such as learning rate and estimators.
    - `num_features`: Numerical features with details on type and constraints.
    - `cat_features`: Categorical features with details on type, allowed values, and constraints.
    - `target`: The target variable for model training (e.g., booking status).
    """

    catalog: str
    db_schema: str = Field(..., alias="schema")
    table_name: str
    parameters: Parameters
    num_features: Dict[str, NumFeature]
    cat_features: Dict[str, CatFeature]
    target: str
