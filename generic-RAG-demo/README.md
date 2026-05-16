# generic-RAG-demo

A generic Retrieval Augmented Generation (RAG) demo from Sogeti Netherlands built in Python. This project demonstrates how to integrate and run different backends, from cloud providers to local models, to parse and process your PDFs, web data, or other text sources.

## Table of Contents

- [generic-RAG-demo](#generic-rag-demo)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Getting started](#getting-started)
    - [Project Environment Setup](#project-environment-setup)
    - [Installation of system dependencies (optional)](#installation-of-system-dependencies-optional)
      - [Unstructered PDF loader](#unstructered-pdf-loader)
      - [Local LLM](#local-llm)
    - [Running generic RAG demo](#running-generic-rag-demo)
    - [config.yaml file](#configyaml-file)
    - [.env file](#env-file)
    - [Chainlit starters](#chainlit-starters)
  - [Dev details](#dev-details)
    - [Linting](#linting)

## Features

- **Multi-backend Support:** Easily switch between cloud-based and local LLMs.
- **Flexible Data Input:** Supports both PDFs and web data ingestion.
- **Configurable Workflows:** Customize settings via a central `config.yaml` file.

## Getting started

### Cloning the repo

Since we're hosting behind a reverse proxy path issues can occur during cloning.
If you're using Git-credential-manager or Oauth to authenticate, you can get a 404 in your browser window.
To resolve, change the oauth callback url (the url you see in your browser 404 page) from frodo.capgemini.com/Oauth/... -> frodo.capgemini.com/gitea/Oauth... (press enter to go to the page)
This will open the authorize page, click 'Authorize' and your clone will continue.

### Azure CLI login troubles
If encountering errors with first time logging into Azure through CLI such as tenant and subscription not showing try the following:

az login --tenant <tenant name>
az login --use-device-code

### Project Environment Setup

This project leverages a modern packaging method defined in `pyproject.toml`. After cloning the repository, you can install the project along with its dependencies. You have two options:

#### Using uv

If you're using uv, simply run the following commands:

```bash
uv venv
uv sync
pip install --upgrade chainlit
uv pip install -e .
```

#### Using a Python Virtual Environment

Alternatively, set up a virtual environment and install the project:

```bash
python -m venv .venv        # Create a new virtual environment named ".venv"
source .venv/bin/activate   # Activate the virtual environment (use ".venv\Scripts\activate" on Windows)
pip install -e .            # Install the project and its dependencies
```

### Installation of system dependencies (optional)

Some optional features require additional system applications to be installed.

#### Unstructered PDF loader

If you would like to run the application using the unstructered PDF loader (`pdf.unstructured` setting) you need to install two system dependencies.

- [poppler-utils](https://launchpad.net/ubuntu/jammy/amd64/poppler-utils)
- [tesseract-ocr](https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract)

```bash
sudo apt install poppler-utils tesseract-ocr
```

> For more information please refer to the [langchain docs.](https://python.langchain.com/docs/integrations/providers/unstructured/)

#### Local LLM

If you would like to run the application using a local LLM backend (`local` settings), you need to install Ollama.

```bash
curl -fsSL https://ollama.com/install.sh | sh  # install Ollama
ollama pull llama3.1:8b  # fetch and download as model
```

Include the downloaded model in the `config.yaml` file:

```yaml
local:
    chat_model: "llama3.1:8b"
    emb_model: "llama3.1:8b"
```

>For more information on installing Ollama, please refer to the Langchain Local LLM documentation, specifically the [Quickstart section](https://python.langchain.com/docs/how_to/local_llms/#quickstart).

### Running generic RAG demo

The `chainlit` app can be ran with the recommended chainlit command.

```bash
python generic_rag/add_sources.py # Add sources to a local chroma database. See config.yaml for specifications

chainlit run generic_rag/app.py # run chainlit app
chainlit run generic_rag/app.py -w # run and reloads the app when module changes

python generic_rag/entry_app.py # run chainlit app with an entry point for (easy) debugging.
```

All configuration options should be set in the `config.yaml` file before hand, environment It is expected at `<project_root>/config.yaml`. Please configure your `config.yaml` and `.env` file with your cloud provider (backend) of choice. See the sections below for more details.

### config.yaml file

A config.yaml file is required to specify your API endpoints and local backends. Use the provided `config.yaml.example` as a starting point. Update the file according to your backend settings and project requirements.

Key configuration points include:

- Chat Backend: Choose among azure, openai, google_vertex, aws, or local.
- Embedding Backend: Configure the embedding models similarly.
- Data Processing Settings: Define PDF and web data sources, chunk sizes, and overlap.
- Vector Database: Customize the path and reset behavior.

For more information on configuring Langchain endpoints and models, please see:

- [langchain cloud chat model doc](https://python.langchain.com/docs/integrations/chat/)
- [langchain local chat model doc](https://python.langchain.com/docs/how_to/local_llms/)
- [langchain cloud/local emb model doc](https://python.langchain.com/docs/integrations/text_embedding/)

> for local models we currently use Ollama

### .env file

Set the API keys for your chosen cloud provider (backend). This ensures that your application can authenticate and interact with the services.

```text
AZURE_OPENAI_API_KEY=your_azure_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Chainlit starters

Chainlit suggestions (starters) can be set with the `CHAINLIT_STARTERS` environment variable.
The variable should be a JSON array of objects with `label` and `message` properties.
An example is as followed.

```text
CHAINLIT_STARTERS=[{"label":"Label 1","message":"Message one."},{"label":"Label 2","message":"Message two."},{"label":"Label 3","message":"Message three."}]
```

## Dev details

### Git

Because there are ongoing limitations with the SSL certificates you can get errors when trying to do simple git actions like push/pull.
`fatal: unable to access 'https://frodo.capgemini.com/gitea/AI_team/generic-RAG-demo.git/': server certificate verification failed. CAfile: none CRLfile: none`
As a workaround, use the `-c http.sslVerify=false` flag in your git commands to circumenvent this issue.
for example: `git push` becomes `git -c http.sslVerify=false`

You can also set it for your repository to automatically add this command by running `git config --local http.sslVerify false` once.

### Install dev dependencies

When you install the dependencies through [uv](#using-uv), the dev dependencies are automatically added.

### pre-commit
pre-commit is used to lint your code and make sure it adheres to the standards.
For first time use, run `pre-commit install` in your terminal to install the pre-commit hooks and make sure it runs everytime you do a commit.

### Linting

Currently [Ruff](https://github.com/astral-sh/ruff) is used as Python linter. It is included in the [pyproject.toml](pyproject.toml) as `dev` dependency if your IDE needs that. However, for VS Code a [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) exists.

### Gitea Workflows

To make sure that our code is linted and tested, we run Gitea workflows (action)  on our pull requests to check.
Without these checks being green, you cannot merge a pull request.

### Common installing errors

#### Chroma hnswlib C++
You get the following error: "error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/"

- **Windows** Follow [this stack overflow post](https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst) to fix it.
- **Linux** follow [this stack overflow post](https://stackoverflow.com/questions/76364672/hnswlib-package-issue-while-installing-chromadb-in-ubuntu) to fix it.


## infra details

prerequisites:

- Follow the Terraform installation instructions [here](https://developer.hashicorp.com/terraform/tutorials/azure-get-started/install-cli)
- Install Azure CLI through the instructions [here](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux?view=azure-cli-latest&pivots=apt)

## Infrastructure as Code (IaC) Structure

This project uses Terraform to provision Azure resources for a generic Retrieval-Augmented Generation (RAG) demo. The codebase is organized for clarity, modularity, and reusability.

### Key Features
- **Input Variables:** All configurable values (location, SKUs, etc.) are defined in `variables.tf` and set in `terraform.auto.tfvars`.
- **Locals:** Frequently reused values and the random suffix for unique resource names are defined in a `locals` block in `main.tf`.
- **Modules:** Role assignments are managed via a reusable module in `modules/role_assignment`.
- **Naming Convention:** Resource names follow Microsoft abbreviation recommendations and use a random 5-letter suffix for global uniqueness where required. See `NAMING.md` for details.

### How to Use Variables
- Edit `terraform.auto.tfvars` to set deployment-specific values.
- See `variables.tf` for descriptions and defaults.

### Modular Structure
- `main.tf`: Core resource definitions and configuration.
- `role-assignments.tf`: Calls the role assignment module for each required assignment.
- `modules/role_assignment`: Contains the reusable role assignment logic.

### Naming Convention
Resource names are constructed as `rag-<abbreviation>[-<suffix>]`, where `<suffix>` is a random 5-letter string for global uniqueness. See `NAMING.md` for more info.
