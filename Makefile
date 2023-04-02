.PHONY : conda
conda :
	conda env create -f environment.yaml --force

.PHONY : config-poetry
config-poetry :
	poetry config experimental.system-git-client true
	poetry config --list

.PHONY : install
install :
	poetry install --no-interaction --all-extras

.PHONY : install-all
install-all :
	poetry install --no-interaction --with exp --all-extras

.PHONY : update
update :
	-poetry self update
	poetry update
	-pre-commit autoupdate

.PHONY : lint
lint :
	ruff check --format=github .

.PHONY : format
format :
	black --check .

.PHONY : unit-test
unit-test :
	python -m pytest --timeout 10 tests/unit

.PHONY : unit-test-cov
unit-test-cov :
	python -m pytest --timeout 10 --cov-report html --cov-report xml --cov-report term --cov=redcat tests/unit

.PHONY : integration-test
integration-test :
	python -m pytest --timeout 60 tests/integration

.PHONY : integration-test-cov
integration-test-cov :
	python -m pytest --timeout 60 --cov-report html --cov-report xml --cov-report term --cov=redcat --cov-append tests/integration

.PHONY : functional-test
functional-test :
	python -m pytest --timeout 60 tests/functional

.PHONY : functional-test-cov
functional-test-cov :
	python -m pytest --timeout 60 --cov-report html --cov-report xml --cov-report term --cov=redcat --cov-append tests/functional

.PHONY : test
make test : unit-test integration-test functional-test

.PHONY : test-cov
make test-cov : unit-test-cov integration-test-cov functional-test-cov

.PHONY : publish-pypi
publish-pypi :
	poetry config pypi-token.pypi ${GRAVITORCH_PYPI_TOKEN}
	poetry publish --build
