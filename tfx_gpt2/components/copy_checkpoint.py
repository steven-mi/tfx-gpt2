import os
import glob
import logging

from shutil import copy2
from tqdm import tqdm

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

        model_dir = get_single_uri(output_dict["model_dir"])
        checkpoint_dir = exec_properties["checkpoint_dir"]
        logging.info("Copying checkpoints from {}".format(checkpoint_dir))
        logging.info("Storing checkpoints to {}".format(model_dir))
        for file in tqdm(glob.glob(os.path.join(checkpoint_dir, "*"))):
            copy2(file, model_dir)


class CopyCheckpointSpec(types.ComponentSpec):
    PARAMETERS = {
        'checkpoint_dir': ExecutionParameter(type=Text),
    }

    INPUTS = {
    }

    OUTPUTS = {
        'model_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }


class CopyCheckpoint(base_component.BaseComponent):
    SPEC_CLASS = CopyCheckpointSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 checkpoint_dir: Text):
        model_dir = external_input("CopyCheckpoint")
        spec = CopyCheckpointSpec(checkpoint_dir=checkpoint_dir,
                                  model_dir=model_dir)

        super(CopyCheckpoint, self).__init__(spec=spec)
