import os
import json
import logging
from tokenizers import ByteLevelBPETokenizer

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
        text_token_size = exec_properties["text_token_size"]
        end_token = exec_properties["end_token"]
        model_dir = get_single_uri(input_dict["model_dir"])
        merged_text_dir = get_single_uri(input_dict["merged_text_dir"])
        encoding_dir = get_single_uri(output_dict["encoding_dir"])

        logging.info("Training BPE Tokenizer")
        tokenizer = ByteLevelBPETokenizer(lowercase=False, end_of_word_suffix=end_token)
        for (dirpath, _, fnames) in os.walk(merged_text_dir):
            for fname in fnames:
                file_path = os.path.join(dirpath, fname)
                if os.path.isfile(file_path):
                    tokenizer.train([file_path], vocab_size=text_token_size)

        encoder_file, vocab_file = tokenizer.save_model(encoding_dir)
        os.rename(encoder_file, os.path.join(encoding_dir, "encoder.json"))
        os.rename(vocab_file, os.path.join(encoding_dir, "vocab.bpe"))
        # load hparams and store with new value
        with open(os.path.join(model_dir, 'hparams.json')) as f:
            hparams = json.load(f)
        hparams["n_vocab"] = text_token_size
        with open(os.path.join(encoding_dir, "hparams.json"), 'w') as json_file:
            json.dump(hparams, json_file)


class CreateEncodingSpec(types.ComponentSpec):
    PARAMETERS = {
        'encoding': ExecutionParameter(type=Text),
        'text_token_size': ExecutionParameter(type=int),
        'end_token': ExecutionParameter(type=Text),

    }

    INPUTS = {
        'model_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'merged_text_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }

    OUTPUTS = {
        'encoding_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }


class CreateEncoding(base_component.BaseComponent):
    SPEC_CLASS = CreateEncodingSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 merged_text_dir: types.Channel,
                 model_dir: types.Channel,
                 end_token: Text,
                 encoding: Text = 'utf-8',
                 text_token_size: int = 50000):
        encoding_dir = external_input("CreateDataset")
        spec = CreateEncodingSpec(model_dir=model_dir,
                                  encoding=encoding,
                                  end_token=end_token,
                                  merged_text_dir=merged_text_dir,
                                  text_token_size=text_token_size,
                                  encoding_dir=encoding_dir)
        super(CreateEncoding, self).__init__(spec=spec)
