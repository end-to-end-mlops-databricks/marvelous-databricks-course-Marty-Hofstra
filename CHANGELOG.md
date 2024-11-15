# Change Log
All notable changes to the `hotel_reservations` project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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
