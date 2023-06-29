# Contributing guidelines

## Before contributing

Welcome to [kerasfuse](https://github.com/ayyucedemirbas/kerasfuse)! Before sending your pull requests, make sure that you __read the whole guidelines__. If you have any doubt on the contributing guide, please feel free to [state it clearly in an issue](https://github.com/ayyucedemirbas/kerasfuse/issues/new)

## Contributing

### Contributor

We are very happy that you are considering implementing algorithms and data structures for others! This repository is referenced and used by learners from all over the globe. Being one of our contributors, you agree and confirm that:

- You did your work - no plagiarism allowed
  - Any plagiarized work will not be merged.
- Your work will be distributed under [GPL-3.0](LICENSE.md) once your pull request is merged
- Your submitted work fulfils or mostly fulfils our styles and standards

__New implementation__ is welcome! For example, new solutions for a problem, different representations for a graph data structure or algorithm designs with different complexity but __identical implementation__ of an existing implementation is not allowed. Please check whether the solution is already implemented or not before submitting your pull request.

__Improving comments__ and __writing proper tests__ are also highly welcome.

### Contribution

We appreciate any contribution, from fixing a grammar mistake in a comment to implementing complex algorithms. Please read this section if you are contributing your work.

Your contribution will be tested by our [automated testing on GitHub Actions](https://github.com/ayyucedemirbas/kerasfuse/actions) to save time and mental energy.  After you have submitted your pull request, you should see the GitHub Actions tests start to run at the bottom of your submission page.  If those tests fail, then click on the ___details___ button try to read through the GitHub Actions output to understand the failure.  If you do not understand, please leave a comment on your submission page and a community member will try to help.

Please help us keep our issue list small by adding fixes: #{$ISSUE_NO} to the commit message of pull requests that resolve open issues. GitHub will use this tag to auto-close the issue when the PR is merged.

# Getting Started with Development

### Cloning the Repository

To start your development journey, you will need to clone the repository from GitHub. Cloning a repository creates a local copy of the project on your machine, allowing you to make changes and contribute to the codebase.

Follow these steps to clone the repository:

1. Open your web browser and navigate to the repository on GitHub.
2. On the repository page, click on the Fork button in the top-right corner. This will create a personal copy of the repository under your GitHub account.
3. Once the repository is forked, go to your GitHub profile and navigate to the forked repository.
4. Click on the green Code button to reveal the cloning options.
5. Copy the URL provided in the cloning options. It should look like https://github.com/your-username/kerasfuse.git.
6. Open your terminal or command prompt on your local machine.
7. Navigate to the directory where you want to clone the repository using the cd command. For example, to navigate to your home directory, you can use cd ~.
8. In the terminal, enter the following command to clone the repository:

```bash
git clone https://github.com/your-username/kerasfuse.git
```
Replace your-username with your GitHub username.

9. Press Enter to execute the command. Git will now download the repository and create a local copy on your machine.

10. Once the cloning process is complete, you can navigate into the cloned repository using the cd command. For example:

```bash
cd kerasfuse
```

Congratulations! You have successfully cloned the repository and are ready to start development. In the next sections, we will cover the setup and configuration steps required for your development environment.


# Setting Up Poetry

To streamline package management and dependency resolution for your project, we will be using Poetry. Poetry is a powerful Python dependency management tool that simplifies the process of managing project dependencies and virtual environments.

Follow these steps to set up Poetry within your project:

1. Ensure that you have Python installed on your local machine. Poetry requires Python 3.6 or higher. You can check your Python version by running the following command in your terminal:

```bash
python --version
```

If Python is not installed or the version is below 3.6, please install or update Python before proceeding.

2. Open your terminal or command prompt and navigate to the root directory of the cloned repository.

3. In the repository's root directory, run the following command to install Poetry: (I assume your are using Linux based system for windows and mac please poetry website for installation)

```bash
curl -sSL https://install.python-poetry.org | python -
```

You should see the installed Poetry version printed in the terminal.

4. Now, let's set up Poetry for your project. Run the following command to initialize kerasfuse project:

#### Tip

If you have multiple Python versions on your system, you can set your Python version by using `poetry env` . Here's an example of how to use it:

```bash
poetry env use python3.10
```
More details at
[poetry-switching-between-environments](https://python-poetry.org/docs/managing-environments/#switching-between-environments)

```bash
poetry install
poetry shell
```

You should now see your command prompt prefixed with (project-name), indicating that you are working within the virtual environment.

Congratulations! You have successfully set up Poetry for your project. You can now manage dependencies, install packages, and run your code within the Poetry environment.

#### Pre-commit plugin
Use [pre-commit](https://pre-commit.com/#installation) to automatically format your code to match our coding style:

```bash
# We assume you complete installation with poetry and you are in the virtualenv created by poetry
pre-commit install # this will create git hook
```

That's it! The plugin will run every time you commit any changes. If there are any errors found during the run, fix them and commit those changes. You can even run the plugin manually on all files:

```bash
pre-commit run --all-files --show-diff-on-failure
```

#### Coding Style

We want your work to be readable by others; therefore, we encourage you to note the following:

- Please write in Python 3.8. For instance:  `print()` is a function in Python 3 so `print "Hello"` will *not* work but `print("Hello")` will.
- Please focus hard on the naming of functions, classes, and variables.  Help your reader by using __descriptive names__ that can help you to remove redundant comments.
  - Single letter variable names are *old school* so please avoid them unless their life only spans a few lines.
  - Expand acronyms because `gcd()` is hard to understand but `greatest_common_divisor()` is not.
  - Please follow the [Python Naming Conventions](https://pep8.org/#prescriptive-naming-conventions) so variable_names and function_names should be lower_case, CONSTANTS in UPPERCASE, ClassNames should be CamelCase, etc.

- We encourage the use of Python [f-strings](https://realpython.com/python-f-strings/#f-strings-a-new-and-improved-way-to-format-strings-in-python) where they make the code easier to read.

## Conventional Commits

To maintain a consistent commit message format and enable automated release management, we follow the Conventional Commits specification. Please adhere to the following guidelines when making commits:

- Use the format: `<type>(<scope>): <description>`

  - `<type>`: Represents the type of change being made. It can be one of the following:
    - **feat**: A new feature
    - **fix**: A bug fix
    - **docs**: Documentation changes
    - **style**: Code style/formatting changes
    - **refactor**: Code refactoring
    - **test**: Adding or modifying tests
    - **chore**: Other changes that don't modify code or test cases

  - `<scope>`: (Optional) Indicates the scope of the change, such as a module or component name.

  - `<description>`: A concise and meaningful description of the change.

- Separate the type, scope, and description with colons and a space.

- Use the imperative mood in the description. For example, "Add feature" instead of "Added feature" or "Adding feature".

- Use present tense verbs. For example, "Fix bug" instead of "Fixed bug" or "Fixes bug".

- Start the description with a capital letter and do not end it with a period.

- If the commit addresses an open issue, include the issue number at the end of the description using the `#` symbol. For example, `fix(user): Resolve login issue #123`.

Example commit messages:

- `feat(user): Add user registration feature`
- `fix(auth): Fix authentication logic`
- `docs(readme): Update project documentation`
- `style(css): Format stylesheets`
- `refactor(api): Simplify API endpoints`
- `test(utils): Add test cases for utility functions`

By following these guidelines, we can maintain a clear and meaningful commit history that helps with code review, collaboration, and automated release processes.