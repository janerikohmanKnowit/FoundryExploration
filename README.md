# Azure AI Foundry exploration helpers

This repository contains a small command-line tool that wraps the Azure CLI to explore Azure AI Foundry resources (hubs and projects). It is designed to work on Windows, macOS, and Linux as long as the Azure CLI is installed and authenticated.

## Prerequisites

1. [Install the Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) and sign in with `az login`.
2. Ensure that the account you are using has sufficient permissions to list Azure AI Foundry resources in the target subscription.
3. (Optional) Activate a Python virtual environment if you plan to extend the tool further.

## Usage

All commands are executed through the module entry point:

```powershell
python -m foundry_cli.cli <command> [options]
```

Replace `python` with `py` on Windows if desired.

### List hub workspaces

```
python -m foundry_cli.cli list-hubs
```

Use `--json` to return the raw API response.

### List project workspaces

```
python -m foundry_cli.cli list-projects
```

### List Foundry projects for a specific hub

```
python -m foundry_cli.cli list-foundry-projects <hub-name> --resource-group <resource-group-name>
```

If you omit `--resource-group`, the tool attempts to infer it from the workspace metadata.

### List everything

```
python -m foundry_cli.cli list-all
```

All commands accept `--subscription <subscription-id>` when you need to override the current Azure CLI context. You can also override the API version with `--api-version` if Microsoft releases a newer API version.

## Extending the tool

The implementation is contained in [`foundry_cli/cli.py`](foundry_cli/cli.py). You can add new subcommands by extending `build_parser` and adding a corresponding branch in `main`.
