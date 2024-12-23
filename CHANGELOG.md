# Change Log
All notable changes to the `hotel_reservations` project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.3.0] - 2024-11-23

### Added
- Implemented Snapshot monitoring class for predictions table to track data drift
- Added drift simulation capability to synthetic data generation
- Added new hotel-reservations-monitoring job with predict and monitor tasks

### Changed
- `generate_synthetic_data` function fully in PySpark, as opposed to switching back and forth to Pandas
- The preprocessor class now has a drift option
- The preprocessor task now has a drift option that is assigned as task parameter in the job config

### Fixed

## [0.2.0] - 2024-11-18

### Added
- Deployment with DAB
- Util functions for the creation of synthetic data (to simulate data ingestion)
- Added the primary key to the projectConfig Pydantic, for addition to the project config yaml file
- Model evaluation task
- Deploy new predictions (as features) task
### Changed
- Moved the creation of ML pipeline preprocessing stages to a model class


### Fixed

## [0.1.0] - 2024-11-10

### Added
- Serving class
- Feature serving task
### Changed
- Featurisation class instead of separate function

### Fixed


## [0.0.2] - 2024-11-06

### Added
- Pytest to the CI pipeline
- Basic model task that is registered in UC models
- Function to register data in the Feature store
- Featurisation task
- Function to retrieve the current branch and SHA in Databricks Git repos
- The Change log
- Example notebook to create Databricks secrets + functionality to retrieve them

### Changed
- `uv sync` for initialisation + the option to switch between test / dev

### Fixed
- Allowed for unit testing with a local SparkSession instead of a remote DatabricksSession

## [0.0.1] - 2024-10-30

### Added
- Data processing function
- Data processing task (i.e. executable)
- Makefile with commands for publishing the .whl to Databricks, and a initialisation function
### Changed

### Fixed
