"""Command-line helpers for exploring Azure AI Foundry resources."""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Optional

AZURE_MGMT_BASE = "https://management.azure.com"
DEFAULT_API_VERSION = "2024-04-01"
DEFAULT_PROJECTS_API_VERSION = "2024-04-01"


class AzureCliError(RuntimeError):
    """Raised when an Azure CLI invocation fails."""


@dataclass
class Workspace:
    """Represents an Azure AI Foundry workspace (hub or project)."""

    id: str
    name: str
    location: str
    kind: Optional[str]
    resource_group: Optional[str]
    raw: dict

    @property
    def is_hub(self) -> bool:
        return (self.kind or "").lower() == "hub"

    @property
    def is_project(self) -> bool:
        return (self.kind or "").lower() == "project"


@dataclass
class FoundryProject:
    """Represents a Foundry project as returned by the management API."""

    id: str
    name: str
    location: Optional[str]
    properties: dict
    raw: dict


class FoundryExplorer:
    """Small helper around the Azure management plane for Foundry resources."""

    def __init__(self, subscription_id: Optional[str] = None, api_version: str = DEFAULT_API_VERSION):
        self.subscription_id = subscription_id or self._get_default_subscription()
        self.api_version = api_version

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_workspaces(self) -> List[Workspace]:
        """Return all Foundry workspaces (hubs & projects) in the subscription."""
        url = f"{AZURE_MGMT_BASE}/subscriptions/{self.subscription_id}/providers/Microsoft.MachineLearningServices/workspaces?api-version={self.api_version}"
        payload = self._az_rest(url)
        value = payload.get("value", [])
        workspaces: List[Workspace] = []
        for item in value:
            properties = item.get("properties", {})
            id_ = item.get("id", "")
            workspace = Workspace(
                id=id_,
                name=item.get("name", ""),
                location=item.get("location", ""),
                kind=properties.get("workspaceType") or properties.get("kind"),
                resource_group=self._extract_resource_group(id_),
                raw=item,
            )
            workspaces.append(workspace)
        return workspaces

    def list_hubs(self) -> List[Workspace]:
        return [ws for ws in self.list_workspaces() if ws.is_hub]

    def list_projects(self) -> List[Workspace]:
        return [ws for ws in self.list_workspaces() if ws.is_project]

    def list_foundry_projects(self, hub_name: str, resource_group: Optional[str] = None, api_version: str = DEFAULT_PROJECTS_API_VERSION) -> List[FoundryProject]:
        """List Foundry projects that live under a hub workspace."""
        resource_group = resource_group or self._get_workspace_resource_group(hub_name)
        url = (
            f"{AZURE_MGMT_BASE}/subscriptions/{self.subscription_id}/resourceGroups/{resource_group}/"
            f"providers/Microsoft.MachineLearningServices/workspaces/{hub_name}/projects?api-version={api_version}"
        )
        payload = self._az_rest(url)
        value = payload.get("value", [])
        projects: List[FoundryProject] = []
        for item in value:
            projects.append(
                FoundryProject(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    location=item.get("location"),
                    properties=item.get("properties", {}),
                    raw=item,
                )
            )
        return projects

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------
    def format_workspaces_table(self, workspaces: Iterable[Workspace]) -> str:
        rows = [(ws.name, ws.kind or "", ws.location, ws.resource_group or "", ws.id) for ws in workspaces]
        headers = ("Name", "Kind", "Location", "Resource Group", "Resource ID")
        return _format_table(headers, rows)

    def format_projects_table(self, projects: Iterable[FoundryProject]) -> str:
        rows = []
        for proj in projects:
            properties = proj.properties
            display_name = properties.get("friendlyName") or properties.get("displayName") or ""
            rows.append((proj.name, display_name, proj.location or "", proj.id))
        headers = ("Name", "Display Name", "Location", "Resource ID")
        return _format_table(headers, rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_default_subscription() -> str:
        result = _run_az(["account", "show", "--query", "id", "-o", "tsv"])
        return result.strip()

    def _get_workspace_resource_group(self, workspace_name: str) -> str:
        workspaces = self.list_workspaces()
        for ws in workspaces:
            if ws.name.lower() == workspace_name.lower():
                if ws.resource_group:
                    return ws.resource_group
                break
        raise ValueError(f"Could not determine resource group for workspace '{workspace_name}'.")

    def _az_rest(self, url: str) -> dict:
        output = _run_az(["rest", "--method", "get", "--url", url])
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            raise AzureCliError(f"Failed to parse response from Azure CLI. Raw output: {output!r}") from exc

    @staticmethod
    def _extract_resource_group(resource_id: str) -> Optional[str]:
        parts = [part for part in resource_id.split("/") if part]
        try:
            index = parts.index("resourceGroups")
            return parts[index + 1]
        except ValueError:
            return None
        except IndexError:
            return None


def _run_az(args: List[str]) -> str:
    cmd = ["az", *args]
    try:
        completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AzureCliError("Azure CLI ('az') is not installed or not on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise AzureCliError(exc.stderr.strip() or str(exc)) from exc
    return completed.stdout


def _format_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    headers = tuple(headers)
    rows = [tuple(str(col) for col in row) for row in rows]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    line_parts = [header.ljust(widths[idx]) for idx, header in enumerate(headers)]
    output_lines = [" | ".join(line_parts), "-+-".join("-" * width for width in widths)]
    for row in rows:
        output_lines.append(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    return "\n".join(output_lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore Azure AI Foundry resources from the CLI.")
    parser.add_argument("--subscription", help="Azure subscription ID. Defaults to the current CLI context.")
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION, help="API version for workspace queries.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    hubs_parser = subparsers.add_parser("list-hubs", help="List hub workspaces in the subscription.")
    hubs_parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")

    projects_parser = subparsers.add_parser("list-projects", help="List project workspaces in the subscription.")
    projects_parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")

    foundry_projects_parser = subparsers.add_parser(
        "list-foundry-projects", help="List Foundry projects underneath a specific hub workspace."
    )
    foundry_projects_parser.add_argument("hub", help="Name of the hub workspace.")
    foundry_projects_parser.add_argument("--resource-group", help="Resource group of the hub workspace.")
    foundry_projects_parser.add_argument(
        "--projects-api-version",
        default=DEFAULT_PROJECTS_API_VERSION,
        help="API version for the Foundry projects management API.",
    )
    foundry_projects_parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")

    all_parser = subparsers.add_parser(
        "list-all", help="List both hub and project workspaces together, grouped by kind."
    )
    all_parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    explorer = FoundryExplorer(subscription_id=args.subscription, api_version=args.api_version)

    if args.command == "list-hubs":
        hubs = explorer.list_hubs()
        if args.json:
            print(json.dumps([hub.raw for hub in hubs], indent=2))
        else:
            print(explorer.format_workspaces_table(hubs))
        return 0

    if args.command == "list-projects":
        projects = explorer.list_projects()
        if args.json:
            print(json.dumps([project.raw for project in projects], indent=2))
        else:
            print(explorer.format_workspaces_table(projects))
        return 0

    if args.command == "list-foundry-projects":
        foundry_projects = explorer.list_foundry_projects(
            args.hub, resource_group=args.resource_group, api_version=args.projects_api_version
        )
        if args.json:
            print(json.dumps([project.raw for project in foundry_projects], indent=2))
        else:
            print(explorer.format_projects_table(foundry_projects))
        return 0

    if args.command == "list-all":
        workspaces = explorer.list_workspaces()
        if args.json:
            print(json.dumps([workspace.raw for workspace in workspaces], indent=2))
        else:
            hubs = [ws for ws in workspaces if ws.is_hub]
            projects = [ws for ws in workspaces if ws.is_project]
            print("Hub Workspaces:")
            print(explorer.format_workspaces_table(hubs) if hubs else "(none)")
            print("\nProject Workspaces:")
            print(explorer.format_workspaces_table(projects) if projects else "(none)")
        return 0

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
