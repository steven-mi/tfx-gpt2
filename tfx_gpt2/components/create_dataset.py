import os
import logging

import numpy as np

from tfx_gpt2.gpt_2 import encoder
from tfx_gpt2.gpt_2.load_dataset import load_dataset

from typing import Any, Dict, List, Text

from tfx import types

from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.components.base import base_executor

from tfx.types import standard_artifacts
from tfx.utils.dsl_utils import external_input

from tfx.types.artifact_utils import get_single_uri

from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter


class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        encoding = exec_properties["encoding"]
        combine = exec_properties["combine"]
        text_path = exec_properties["text_path"]
        model_path = get_single_uri(input_dict["model_path"])
        dataset_path = os.path.join(get_single_uri(output_dict["dataset_path"]), "dataset.npz")

        enc = encoder.get_encoder(model_path)
        logging.info('Reading files')
        chunks = load_dataset(enc, text_path, combine, encoding=encoding)
        logging.info('Writing', dataset_path)
        np.savez_compressed(dataset_path, *chunks)


class CreateDatasetSpec(types.ComponentSpec):
    PARAMETERS = {
        'encoding': ExecutionParameter(type=Text),
        'combine': ExecutionParameter(type=int),
        'text_path': ExecutionParameter(type=Text),
    }

    INPUTS = {
        'model_path': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }

    OUTPUTS = {
        'dataset_path': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }


class CreateDataset(base_component.BaseComponent):
    SPEC_CLASS = CreateDatasetSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 text_path: Text,
                 model_path: types.Channel,
                 encoding: Text = 'utf-8',
                 combine: int = 50000):
        dataset_path = external_input("CreateDataset")
        spec = CreateDatasetSpec(text_path=text_path,
                                 model_path=model_path,
                                 encoding=encoding,
                                 combine=combine,
                                 dataset_path=dataset_path)
        super(CreateDataset, self).__init__(spec=spec)
