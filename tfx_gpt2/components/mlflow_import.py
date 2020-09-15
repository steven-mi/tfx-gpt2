import os
import pickle
import logging

import mlflow

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


class MLFlowImportSpec(types.ComponentSpec):
    PARAMETERS = {
        'model_name': ExecutionParameter(type=Text),
        'mlflow_tracking_url': ExecutionParameter(type=int),
        'mlflow_host': ExecutionParameter(type=int),
        'mlflow_port': ExecutionParameter(type=int),
    }

    INPUTS = {
        'model_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'artifact_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'hyperparameter_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'metric_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }

    OUTPUTS = {
    }


class MLFlowImport(base_component.BaseComponent):
    SPEC_CLASS = MLFlowImportSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 model_name: Text,
                 mlflow_tracking_url: Text,
                 mlflow_host: Text,
                 mlflow_port: Text,
                 model_dir: types.Channel,
                 artifact_dir: types.Channel,
                 hyperparameter_dir: types.Channel,
                 metric_dir: types.Channel):
        spec = MLFlowImportSpec(model_name=model_name,
                                mlflow_tracking_url=mlflow_tracking_url,
                                mlflow_host=mlflow_host,
                                mlflow_port=mlflow_port,
                                model_dir=model_dir,
                                artifact_dir=artifact_dir,
                                hyperparameter_dir=hyperparameter_dir,
                                metric_dir=metric_dir)
        super(MLFlowImport, self).__init__(spec=spec)
