from typing import List, Literal, TypedDict, Union


class Constraints(TypedDict, total=False):
    """
    Specifies optional constraints for a feature, such as minimum or maximum values.
    - `min`: The minimum allowed value for the feature.
    - `max`: The maximum allowed value for the feature.
    """

    min: Union[int, float]
    max: Union[int, float]


class NumFeature(TypedDict):
    """
    Describes a numerical feature in the dataset.
    - `type`: Indicates the data type, either 'integer' or 'float'.
    - `constraints`: Optional constraints for the numerical feature, like `min` or `max`.
    """

    type: Literal["integer", "float"]
    constraints: Constraints


class CatFeature(TypedDict, total=False):
    """
    Describes a categorical feature in the dataset.
    - `type`: Indicates the data type, either 'string' or 'bool'.
    - `allowed_values`: Lists permissible values for string features, if any.
    - `constraints`: Optional constraints for integer-based categorical values.
    """

    type: Literal["string", "bool"]
    allowed_values: List[str]
    constraints: Constraints


class Parameters(TypedDict):
    """
    Holds model parameters.
    - `learning_rate`: Learning rate for the model (e.g., 0.01).
    - `n_estimators`: Number of estimators (e.g., 1000).
    - `max_depth`: Maximum tree depth (e.g., 6).
    """

    learning_rate: float
    n_estimators: int
    max_depth: int


class ProjectConfigType(TypedDict):
    """
    Defines the configuration for the project.
    - `catalog`: The data catalog name.
    - `schema`: The schema where the dataset resides.
    - `table_name`: The table name of the dataset.
    - `parameters`: Model parameters such as learning rate and estimators.
    - `num_features`: Numerical features with details on type and constraints.
    - `cat_features`: Categorical features with details on type, allowed values, and constraints.
    - `target`: The target variable for model training (e.g., booking status).
    """

    catalog: str
    schema: str
    table_name: str
    parameters: Parameters
    num_features: dict[str, NumFeature]
    cat_features: dict[str, CatFeature]
    target: str
