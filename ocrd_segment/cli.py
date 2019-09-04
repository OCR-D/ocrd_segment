import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_segment.repair import RepairSegmentation
from ocrd_segment.evaluate import EvaluateSegmentation

@click.command()
@ocrd_cli_options
def ocrd_segment_repair(*args, **kwargs):
    return ocrd_cli_wrap_processor(RepairSegmentation, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_segment_evaluate(*args, **kwargs):
    return ocrd_cli_wrap_processor(EvaluateSegmentation, *args, **kwargs)
