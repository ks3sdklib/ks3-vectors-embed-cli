#!/usr/bin/env python3
"""Main CLI entry point for KS3 Vectors."""

import click
from rich.console import Console
from rich.traceback import install

from ks3_vectors.commands.embed_put import embed_put
from ks3_vectors.commands.embed_query import embed_query
from ks3_vectors.__version__ import __version__

# Install rich traceback handler
install(show_locals=True)
console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="ks3-vectors-embed")
@click.option('--base-url', help='Embedding API base URL to use')
@click.option('--endpoint', help='KS3 object-storage host to use')
@click.option('--vectors-endpoint', help='KS3 vector-bucket host to use')
@click.option('--debug', is_flag=True, help='Enable debug mode with detailed logging')
@click.pass_context
def cli(ctx, base_url, endpoint, vectors_endpoint, debug):
    """KS3 Vectors Embed CLI - Standalone tool for vector embedding operations with KS3."""
    ctx.ensure_object(dict)
    ctx.obj['console'] = console
    ctx.obj['debug'] = debug
    ctx.obj['base_url'] = base_url
    ctx.obj['endpoint'] = endpoint
    ctx.obj['vectors_endpoint'] = vectors_endpoint

    if debug:
        console.print("[dim] Debug mode enabled[/dim]")


# Register commands as subcommands
cli.add_command(embed_put, name='put')
cli.add_command(embed_query, name='query')


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise


if __name__ == '__main__':
    main()
