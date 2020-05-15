import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from .repair import RepairSegmentation
from .import_image_segmentation import ImportImageSegmentation
from .import_coco_segmentation import ImportCOCOSegmentation
from .evaluate import EvaluateSegmentation
from .extract_pages import ExtractPages
from .extract_regions import ExtractRegions
from .extract_lines import ExtractLines
from .extract_address import ExtractAddress
from .detect_address import DetectAddress

@click.command()
@ocrd_cli_options
def ocrd_segment_repair(*args, **kwargs):
    return ocrd_cli_wrap_processor(RepairSegmentation, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_segment_from_masks(*args, **kwargs):
    return ocrd_cli_wrap_processor(ImportImageSegmentation, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_segment_from_coco(*args, **kwargs):
    return ocrd_cli_wrap_processor(ImportCOCOSegmentation, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_segment_evaluate(*args, **kwargs):
    return ocrd_cli_wrap_processor(EvaluateSegmentation, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_segment_extract_pages(*args, **kwargs):
    return ocrd_cli_wrap_processor(ExtractPages, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_segment_extract_regions(*args, **kwargs):
    return ocrd_cli_wrap_processor(ExtractRegions, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_segment_extract_lines(*args, **kwargs):
    return ocrd_cli_wrap_processor(ExtractLines, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_segment_extract_address(*args, **kwargs):
    return ocrd_cli_wrap_processor(ExtractAddress, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_segment_detect_address(*args, **kwargs):
    return ocrd_cli_wrap_processor(DetectAddress, *args, **kwargs)
