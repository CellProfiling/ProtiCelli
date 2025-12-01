# NEW_REPO

This repository serves only as a Python template for new projects.

## Create a new repository

- Create a [new repo](https://github.com/new) and select `CellProfiling/cell-pro-template` as template repository.
- Clone your new repo.
- Search and replace all occurences of `NEW_REPO`, `AUTHOR_NAME` and `AUTHOR_EMAIL`. Replace `NEW_REPO` with the name of the new repo.
- Add package requirements in `install_requires` in [`setup.py`](setup.py) and in [`requirements.txt`](requirements.txt) as needed.
- Update this `README.md` with a description of and instructions for your new repo.

## Development

- Install and set up development environment.

  ```sh
  pip install -r requirements_dev.txt
  ```

  This will install all requirements.
It will also install this package in development mode, so that code changes are applied immediately without reinstall necessary.

- Here's a list of development tools we use.
  - [black](https://pypi.org/project/black/)
  - [flake8](https://pypi.org/project/flake8/)
  - [pydocstyle](https://pypi.org/project/pydocstyle/)
  - [pylint](https://pypi.org/project/pylint/)
  - [pytest](https://pypi.org/project/pytest/)
  - [tox](https://pypi.org/project/tox/)
- It's recommended to use the corresponding code formatter and linters also in your code editor to get instant feedback. A popular editor that can do this is [`vscode`](https://code.visualstudio.com/).
- Run all tests, check formatting and linting.

  ```sh
  tox
  ```

- Run a single tox environment.

  ```sh
  tox -e lint
  ```

- Reinstall all tox environments.

  ```sh
  tox -r
  ```

- Run pytest and all tests.

  ```sh
  pytest
  ```

- Run pytest and calculate coverage for the package.

  ```sh
  pytest --cov-report term-missing --cov=NEW_REPO
  ```

- Continous integration is by default supported via [GitHub actions](https://help.github.com/en/actions). GitHub actions is free for public repos and comes with 2000 free Ubuntu build minutes per month for private repos.

- To activate continuous integration testing on Travis CI, add a `.travis.yml` file with this contents to the repo.

  ```yaml
  dist: xenial
  language: python
  cache: pip
  python:
    - "3.6"
    - "3.7"
    - "3.8"
  install:
    - pip install -U tox-travis
  script: tox
  ```

  Note that Travis CI is free for public repos, but requires a subscription for private repos.
