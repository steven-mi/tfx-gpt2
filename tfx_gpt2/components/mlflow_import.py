import os
import glob
import pickle
import mlflow
import mlflow.tensorflow
import tensorflow as tf

from typing import Any, Dict, List, Text

from tfx import types

from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.components.base import base_executor

from tfx.types import standard_artifacts

from tfx.types.artifact_utils import get_single_uri

from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter


class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        model_name = exec_properties["model_name"]
        mlflow_tracking_url = exec_properties["mlflow_tracking_url"]

        model_dir = get_single_uri(input_dict["model_dir"])
        artifact_dir = get_single_uri(input_dict["artifact_dir"])
        hyperparameter_dir = get_single_uri(input_dict["hyperparameter_dir"])
        metric_dir = get_single_uri(input_dict["metric_dir"])

        mlflow.set_tracking_uri(mlflow_tracking_url)
        mlflow.set_experiment(model_name)
        with mlflow.start_run():
            with open(glob.glob(os.path.join(hyperparameter_dir, "*.pickle"))[0], 'rb') as fp:
                hyperparameter = pickle.load(fp)
                for k, v in hyperparameter.items():
                    mlflow.log_param(k, v)
            with open(glob.glob(os.path.join(metric_dir, "*.pickle"))[0], 'rb') as fp:
                metric = pickle.load(fp)
                for k, v in metric.items():
                    mlflow.log_metric(k, v)
            for artifact in glob.glob(os.path.join(artifact_dir, "*")):
                mlflow.log_artifact(artifact)
            with open(glob.glob(os.path.join(model_dir, "*.pickle"))[0], 'rb') as fp:
                mlflow.tensorflow.log_model(tf_saved_model_dir=model_dir, tf_meta_graph_tags=["serve"],
                                            tf_signature_def_key="predict", artifact_path="GPT2")


class MLFlowImportSpec(types.ComponentSpec):
    PARAMETERS = {
        'model_name': ExecutionParameter(type=Text),
        'mlflow_tracking_url': ExecutionParameter(type=Text),
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
                 model_dir: types.Channel,
                 artifact_dir: types.Channel,
                 hyperparameter_dir: types.Channel,
                 metric_dir: types.Channel):
        spec = MLFlowImportSpec(model_name=model_name,
                                mlflow_tracking_url=mlflow_tracking_url,
                                model_dir=model_dir,
                                artifact_dir=artifact_dir,
                                hyperparameter_dir=hyperparameter_dir,
                                metric_dir=metric_dir)
        super(MLFlowImport, self).__init__(spec=spec)
