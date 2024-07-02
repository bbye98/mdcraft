# MDCraft Contribution Guidelines

Thank you for your interest in contributing to MDCraft, a comprehensive Python package for streamlining molecular dynamics (MD) simulation workflows. We welcome contributions from the community to improve and expand the functionality of MDCraft.

## Code of conduct

By participating in this project, you agree to abide by the [Contributor Covenant](CODE_OF_CONDUCT.md). Please be respectful and considerate in your interactions with others.

## How to contribute

To get an overview of the project, read the [README](README.md) file.

There are several ways you can contribute to MDCraft, including but not limited to

* asking and answering questions in [discussions](https://github.com/bbye98/mdcraft/discussions),
* reporting bugs and requesting features by submitting new issues,
* adding new features and fixing bugs by creating pull requests (PRs),
* improving and maintaining consistency in the documentation by updating numpydoc-style docstrings, and
* providing reproducible examples and tutorials in Jupyter notebooks.

## Getting started

### Issues

#### Open a new issue

Before reporting a bug or requesting a feature, search to see if a related issue already exists. If the results comes up empty, you can [submit a new issue](https://github.com/bbye98/mdcraft/issues/new). Make sure you include a clear and descriptive title and provide as much detail as possible to help us understand and reproduce the issue.

#### Solve an issue

Scan through our existing issues to find one that interests you. You can narrow down the search using the labels as filters. If you find an issue to work on, you are welcome to open a PR with a fix.

### Make changes

To contribute to MDCraft, you must follow the "fork and pull request" workflow below.

1. [Fork the repository.](https://github.com/bbye98/mdcraft/fork)
2. Clone the fork to your machine using Git and change to the directory:

       git clone https://github.com/<your-github-username>/mdcraft.git
       cd mdcraft

3. Create a new branch and check it out:

       git checkout -b <branch-name>

4. Start working on your changes! You may want to create and activate an environment, and then install all dependencies:

       python3 -m pip install -r requirements.txt

Remember to

* write clean and readable code by following [PEP 8](https://peps.python.org/pep-0008/) style guidelines,
* ensure docstrings adhere to the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style guidelines, and
* add Pytest-based unit tests for new features and bug fixes.

### Commit your update

When you are ready to submit your changes to GitHub, follow the steps below.

1. Ensure that your local copy of MDCraft passes all the unit tests, including any that you may have written, using pytest.
2. Stage and commit your local files.

       git add .
       git commit -m "<short-description-of-your-changes>

3. Push changes to the `<branch-name>` branch of your GitHub fork of MDCraft.

       git push

### Pull request

If you wish to contribute your changes to the main MDCraft project, [make a PR](https://github.com/bbye98/mdcraft/compare). The project maintainers will review your PR and, if it provides a significant or useful change to MDCraft, will be merged!