# -*- coding: utf-8 -*-
import logging

import click


@click.command(short_help='Tests a model')
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('feature_extractor_path', type=click.Path(exists=True))
@click.argument('testset_path', type=click.Path(exists=True))
@click.argument('predictions_path', type=click.Path())
@click.argument('report_path', type=click.Path())
def test_model(model_path, feature_extractor_path, testset_path,
               predictions_path, report_path):
    """Tests a model.

    Reads a model from MODEL_PATH, a feature extractor pipeline from
    FEATURE_EXTRACTOR_PATH, and a test set from TESTSET_PATH in order to make
    predictions on it and run the metrics.

    It then saves the predictions in PREDICTIONS_PATH and the metrics report in
    REPORT_PATH.
    """
    logger = logging.getLogger(__name__)
    logger.info('reading model from {}'.format(model_path))
    logger.info('reading feature extractor pipeline from {}'.format(
        feature_extractor_path))
    logger.info('reading test dataset from {}'.format(
        testset_path))
    logger.info('saving predictions in {}'.format(predictions_path))
    logger.info('saving report in {}'.format(report_path))
