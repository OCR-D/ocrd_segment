import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_evaluate_segmentation.evaluate import EvaluateSegmentation

@click.command()
@ocrd_cli_options
def ocrd_evaluate_segmentation(*args, **kwargs):
    return ocrd_cli_wrap_processor(EvaluateSegmentation, *args, **kwargs)
