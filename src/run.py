#!/usr/bin/env python2

"""
Entry point for command line execution of project.
"""
import click
import yaml

from evaluation import evaluate

COMMAND_HANDLERS = {'eval': evaluate.evaluate}

@click.command()
@click.argument('command', type=click.Choice(COMMAND_HANDLERS.keys()))
@click.argument('config', type=click.Path(exists=True))
def main(command, config):
    # Load config file
    with open(config, 'r') as fin:
        config = yaml.safe_load(fin)

    # Call handler
    handler = COMMAND_HANDLERS(command)
    handler(config)

if __name__=='__main__':
    main()
