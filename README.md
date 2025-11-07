# Azure AI Foundry exploration helpers

This repository contains a small command-line tool that wraps the Azure CLI to explore Azure AI Foundry resources (hubs and projects). It is designed to work on Windows, macOS, and Linux as long as the Azure CLI is installed and authenticated. The helper automatically handles paginated responses from the management API and reuses data during a single run to keep Azure CLI traffic minimal.

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

### Common options

All commands accept the following options:

* `--subscription <id>` – override the Azure subscription from your current CLI context.
* `--api-version <version>` – override the management API version used for workspace queries.

Several commands also accept `--resource-group <name>` to scope requests to a single resource group (helpful for large subscriptions).

### List hub workspaces

```
python -m foundry_cli.cli list-hubs [--resource-group <name>] [--json]
```

Use `--json` to return the raw API response.

### List project workspaces

```
python -m foundry_cli.cli list-projects [--resource-group <name>] [--json]
```

### List Foundry projects for a specific hub

```
python -m foundry_cli.cli list-foundry-projects <hub-name> [--resource-group <resource-group-name>] [--json]
```

If you omit `--resource-group`, the tool attempts to infer it from the workspace metadata.

To iterate across every hub in the subscription (or a single resource group), use:

```
python -m foundry_cli.cli list-foundry-projects --all [--resource-group <name>] [--json]
```

When running with table output, hubs without projects are summarized after the table to make it easier to identify empty hubs. JSON output includes the hub name for each group of projects to simplify further automation.

Use `--projects-api-version` if you need to pin a specific preview or GA version of the Foundry projects API.

### Summarize Foundry roots and projects

```
python -m foundry_cli.cli list-foundry-inventory [--resource-group <name>] [--roots-only] [--json]
```

This command produces a compact inventory that highlights the two Azure AI Foundry root types—**Foundry Hubs (v1)** and **Foundry Resources (v2)**—and the projects that live beneath them. Each root appears at least once with a placeholder (`-`) in the project column so you can quickly distinguish the container rows from project entries. Use `--roots-only` when you only need the containers without enumerating the child projects. `--projects-api-version` controls the API version used to enumerate projects for both hub and resource roots.

### List everything

```
python -m foundry_cli.cli list-all [--resource-group <name>] [--json]
```

All commands accept `--subscription <subscription-id>` when you need to override the current Azure CLI context. You can also override the API version with `--api-version` if Microsoft releases a newer API version.

### Show workspace details

```
python -m foundry_cli.cli show-workspace <name> [--resource-group <resource-group-name>] [--json]
```

This command is useful when you need the full resource payload for a single hub or project workspace without listing the entire subscription.

## Extending the tool

The implementation is contained in [`foundry_cli/cli.py`](foundry_cli/cli.py). You can add new subcommands by extending `build_parser` and adding a corresponding branch in `main`.
