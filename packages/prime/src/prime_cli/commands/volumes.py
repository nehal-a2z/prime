"""`prime volumes`: named volumes for hosted training data and outputs.

A volume is a PVC in your team's (or personal) namespace on the cluster it
was created on. `prime train config.toml --volume <name>` makes the run
write under `runs/<runId>/` on it, and it outlives every run. For hosted
SFT, `prime volumes stage` puts an HF dataset on it under
`datasets/<name>` so the run can read `[data] name = "/datasets/<name>"`.
"""

from typing import Optional

import typer
from rich.table import Table

from prime_cli.api.training import HostedTrainingClient
from prime_cli.core import APIClient, APIError, Config

from ..utils import (
    PlainTyper,
    confirm_or_skip,
    get_console,
    output_data_as_json,
    validate_output_format,
)
from ..utils.env_vars import EnvParseError
from .volumes_stage import StageError, print_stage_output, resolve_hf_token, stage_dataset

app = PlainTyper(
    help="Manage volumes for full-FT run outputs (closed beta)",
    no_args_is_help=True,
)
console = get_console()


def _client() -> tuple[HostedTrainingClient, str | None]:
    return HostedTrainingClient(APIClient()), Config().team_id


@app.command()
def create(
    name: str = typer.Argument(..., help="Volume name (lowercase letters, digits, '-')"),
    size: str = typer.Option("1Ti", "--size", help="Size, e.g. 500Gi or 2Ti. Can grow later."),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """Create a volume on your team's (or personal) cluster."""
    validate_output_format(output, console)
    client, team_id = _client()
    try:
        volume = client.create_volume(name, size, team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if output == "json":
        output_data_as_json(volume.model_dump(by_alias=True), console)
        return
    console.print(f"[green]Creating volume {volume.name} ({volume.size}).[/green]")
    console.print(f"Use it with: prime train config.toml --volume {volume.name}")


@app.command("list")
def list_volumes(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table or json"),
) -> None:
    """List your volumes."""
    validate_output_format(output, console)
    client, team_id = _client()
    try:
        volumes = client.list_volumes(team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if output == "json":
        output_data_as_json([v.model_dump(by_alias=True) for v in volumes], console)
        return
    table = Table("Name", "Size", "Status", "Namespace", "Created")
    for v in volumes:
        table.add_row(v.name, v.size or "-", v.status, v.namespace, v.created_at or "-")
    console.print(table)


@app.command()
def stage(
    source: str = typer.Argument(
        ...,
        help=(
            "HF dataset repository ID, 'name' or 'owner/name' (no URLs, "
            "hf:// aliases, or local paths - local upload is deferred; publish "
            "a dataset repo first)"
        ),
    ),
    volume: str = typer.Option(..., "--volume", help="Existing volume to stage onto."),
    path: Optional[str] = typer.Option(
        None,
        "--path",
        help=(
            "Dataset directory name under datasets/ (default: repository "
            "basename, kept exactly). One directory component; never sanitized."
        ),
    ),
    namespace: str = typer.Option(
        "auto",
        "--namespace",
        help="Must be 'auto' (use the volume's API-returned namespace) or match it.",
    ),
    kube_context: str = typer.Option(
        None,
        "--kube-context",
        help=(
            "Kubeconfig context to use. Defaults to the current context; the "
            "context must point at the volume's cluster."
        ),
    ),
    revision: str = typer.Option(
        "main", "--revision", help="HF revision (branch, tag, or SHA) to stage."
    ),
    env_file: list[str] = typer.Option(
        [], "--env-file", help="Env file that may contain HF_TOKEN (other entries ignored)."
    ),
    env_var: list[str] = typer.Option(
        [], "--env-var", "-e", help="Set HF_TOKEN (e.g. -e HF_TOKEN=hf_...)."
    ),
    timeout_seconds: int = typer.Option(
        3600, "--timeout-seconds", help="Overall staging deadline in seconds."
    ),
    output: str = typer.Option(
        "table", "--output", "-o", help="Output format: table or json (json: progress on stderr)."
    ),
) -> None:
    """Stage an HF dataset repository onto a volume for hosted SFT.

    Downloads the dataset snapshot inside a short-lived CPU pod on the
    volume's cluster, verifies it offline in a fresh process, then
    publishes it atomically under datasets/<name>. Requires kubectl and
    an authorized kubeconfig context for the volume's namespace (pods,
    pod logs, PVC read; plus Secrets for private datasets).

    Recipe:

    \b
        prime volumes create research
        prime volumes stage PrimeIntellect/INTELLECT-3-SFT-10K --volume research
        # then in your SFT TOML:  [data]
        #     type = "sft"
        #     name = "/datasets/intellect-3-sft-10k"
        #     splits = ["math"]
        prime train sft.toml --volume research

    Staged datasets are immutable: re-staging the same revision is
    idempotent ('already staged'); changed upstream content needs a new
    --path. Local laptop uploads are deferred in v1.
    """
    validate_output_format(output, console)
    client, team_id = _client()
    try:
        token = resolve_hf_token(list(env_var), list(env_file))
    except EnvParseError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except StageError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    json_mode = output == "json"
    try:
        result = stage_dataset(
            client=client,
            team_id=team_id,
            source=source,
            volume_name=volume,
            path=path,
            namespace=namespace,
            kube_context=kube_context,
            revision=revision,
            token=token,
            timeout_seconds=timeout_seconds,
            json_mode=json_mode,
        )
    except StageError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        raise typer.Exit(130)

    print_stage_output(result, json_mode)


@app.command()
def resize(
    name: str = typer.Argument(..., help="Volume name"),
    size: str = typer.Option(..., "--size", help="New size, larger than the current one"),
) -> None:
    """Grow a volume in place. Running pods see the new size; volumes can't shrink."""
    client, team_id = _client()
    try:
        volume = client.resize_volume(name, size, team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]Volume {volume.name} is now {volume.size}.[/green]")


@app.command()
def delete(
    name: str = typer.Argument(..., help="Volume name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a volume and everything on it."""
    if not confirm_or_skip(f"Delete volume {name} and all run data on it?", yes):
        raise typer.Exit(0)
    client, team_id = _client()
    try:
        client.delete_volume(name, team_id=team_id)
    except APIError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(f"[green]Deleting volume {name}.[/green]")
