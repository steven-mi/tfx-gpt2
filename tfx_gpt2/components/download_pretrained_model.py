import os
import requests
import logging

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
        model_name = exec_properties["model_name"]
        logging.info("Downloading pretrained model of {}".format(model_name))
        logging.info("Storing pretrained mdoel to {}".format(model_dir))

        subdir = os.path.join('models', model_name)
        subdir = subdir.replace('\\', '/')  # needed for Windows
        for filename in ['checkpoint', 'encoder.json', 'hparams.json', 'model.ckpt.data-00000-of-00001',
                         'model.ckpt.index', 'model.ckpt.meta', 'vocab.bpe']:
            logging.info("Getting {}".format(filename))
            # get file from storage server
            r = requests.get("https://storage.googleapis.com/gpt-2/" + subdir + "/" + filename, stream=True)
            # save to output path
            with open(os.path.join(model_dir, filename), 'wb') as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)


class DownloadPretrainedModelSpec(types.ComponentSpec):
    PARAMETERS = {
        'model_name': ExecutionParameter(type=Text),
    }

    INPUTS = {
    }

    OUTPUTS = {
        'model_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }


class DownloadPretrainedModel(base_component.BaseComponent):
    SPEC_CLASS = DownloadPretrainedModelSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 model_name: Text = "117M"):
        model_dir = external_input("DownloadPretrainedModel")
        spec = DownloadPretrainedModelSpec(model_name=model_name,
                                           model_dir=model_dir)

        super(DownloadPretrainedModel, self).__init__(spec=spec)
