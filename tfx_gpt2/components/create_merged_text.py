import os
import json
import logging

import pandas as pd

from typing import Any, Dict, List, Text, Union, Optional

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
        end_token = exec_properties["end_token"]

        text_dir = exec_properties["text_dir"]
        merged_text_dir = get_single_uri(output_dict["merged_text_dir"])
        merged_text_path = os.path.join(merged_text_dir, "merged_text")

        raw_text = ''
        for (dirpath, _, fnames) in os.walk(text_dir):
            for fname in fnames:
                file_path = os.path.join(dirpath, fname)

                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    for index, row in df.iterrows():
                        raw_text += row["text"] + end_token
                elif "wiki" in fname:
                    for line in open(file_path, 'r'):
                        raw_text += json.loads(line)["text"] + end_token
                else:
                    # Plain text
                    with open(file_path, 'r', encoding=encoding) as fp:
                        raw_text += fp.read() + end_token

                with open(merged_text_path, "a") as text_file:
                    text_file.write(raw_text)
                raw_text = ''
        logging.info("Saved merged text to {}".format(merged_text_dir))
        return 0


class CreateMergedTextSpec(types.ComponentSpec):
    PARAMETERS = {
        'encoding': ExecutionParameter(type=Text),
        'text_dir': ExecutionParameter(type=Text),
        'end_token': ExecutionParameter(type=Text),
    }

    INPUTS = {
    }

    OUTPUTS = {
        'merged_text_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }


class CreateMergedText(base_component.BaseComponent):
    SPEC_CLASS = CreateMergedTextSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 end_token: Text,
                 text_dir: Optional[Text] = None,
                 encoding: Text = 'utf-8'):
        merged_text_dir = external_input("CreateDataset")

        spec = CreateMergedTextSpec(text_dir=text_dir,
                                    end_token=end_token,
                                    encoding=encoding,
                                    merged_text_dir=merged_text_dir)

        super(CreateMergedText, self).__init__(spec=spec)
