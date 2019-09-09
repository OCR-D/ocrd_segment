import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_segment.repair import RepairSegmentation
from ocrd_segment.evaluate import EvaluateSegmentation
from ocrd_segment.extract_gt import ExtractGT

@click.command()
@ocrd_cli_options
def ocrd_segment_repair(*args, **kwargs):
    return ocrd_cli_wrap_processor(RepairSegmentation, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_segment_evaluate(*args, **kwargs):
    return ocrd_cli_wrap_processor(EvaluateSegmentation, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_segment_extract_gt(*args, **kwargs):
    return ocrd_cli_wrap_processor(ExtractGT, *args, **kwargs)
