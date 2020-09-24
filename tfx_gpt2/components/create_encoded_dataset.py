import os
import logging

import numpy as np

from tfx_gpt2.core import encoder
from tfx_gpt2.core.load_dataset import load_dataset

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
        merged_text_dir = get_single_uri(input_dict["merged_text_dir"])
        encoding_dir = get_single_uri(input_dict["encoding_dir"])
        end_token = exec_properties["end_token"]

        logging.info('Reading files')
        enc = encoder.get_encoder(encoding_dir)
        chunks = load_dataset(enc, merged_text_dir, encoding=encoding, end_token=end_token)

        dataset_path = os.path.join(get_single_uri(output_dict["dataset_dir"]), "dataset.npz")
        logging.info('Writing', dataset_path)
        np.savez_compressed(dataset_path, *chunks)


class CreateEncodedDatasetSpec(types.ComponentSpec):
    PARAMETERS = {
        'encoding': ExecutionParameter(type=Text),
        'end_token': ExecutionParameter(type=Text),
    }

    INPUTS = {
        'merged_text_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'encoding_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }

    OUTPUTS = {
        'dataset_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }


class CreateEncodedDataset(base_component.BaseComponent):
    SPEC_CLASS = CreateEncodedDatasetSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 merged_text_dir: types.Channel,
                 encoding_dir: types.Channel,
                 end_token: Text,
                 encoding: Text = 'utf-8'):
        dataset_dir = external_input("CreateDataset")
        spec = CreateEncodedDatasetSpec(merged_text_dir=merged_text_dir,
                                        encoding_dir=encoding_dir,
                                        encoding=encoding,
                                        end_token=end_token,
                                        dataset_dir=dataset_dir)
        super(CreateEncodedDataset, self).__init__(spec=spec)
