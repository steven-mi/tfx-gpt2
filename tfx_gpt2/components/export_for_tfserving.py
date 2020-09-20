import os
import json
import shutil
import numpy as np
import tensorflow as tf

from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from tfx_gpt2.core import sample, model

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


def export_for_serving(encoding_dir, checkpoint_dir, export_dir, train_config, seed=0):
    hparams = model.default_hparams()
    with open(os.path.join(encoding_dir, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    length = hparams.n_ctx

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [train_config["batch_size"], None])
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams,
            length=length,
            context=context,
            batch_size=train_config["batch_size"],
            temperature=1.0,
            top_k=train_config["top_k"]
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, ckpt)

        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        signature = predict_signature_def(inputs={'context': context},
                                          outputs={'sample': output})
        builder.add_meta_graph_and_variables(sess,
                                             [tf.saved_model.SERVING],
                                             signature_def_map={"predict": signature},
                                             strip_default_attrs=True)
        builder.save()


class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        train_config = exec_properties["train_config"]

        checkpoint_dir = get_single_uri(input_dict["checkpoint_dir"])
        encoding_dir = get_single_uri(input_dict["encoding_dir"])

        export_dir = get_single_uri(output_dict["export_dir"])

        # copy encodings to export dir because they are needed for inference!
        src_files = os.listdir(encoding_dir)
        for file_name in src_files:
            full_file_name = os.path.join(encoding_dir, file_name)
            if 'encoder.json' in full_file_name or 'hparams.json' in full_file_name or 'vocab.bpe' in full_file_name:
                shutil.copy(full_file_name, export_dir)

        export_for_serving(encoding_dir=encoding_dir,
                           checkpoint_dir=checkpoint_dir,
                           export_dir=export_dir,
                           train_config=train_config)


class ExportToTFServingSpec(types.ComponentSpec):
    PARAMETERS = {
        'train_config': ExecutionParameter(type=Dict),

    }

    INPUTS = {
        'checkpoint_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'encoding_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }

    OUTPUTS = {
        'export_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }


class ExportToTFServing(base_component.BaseComponent):
    SPEC_CLASS = ExportToTFServingSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 checkpoint_dir: types.Channel,
                 encoding_dir: types.Channel,
                 train_config: Dict):
        export_dir = external_input("ExportToTFServing")

        spec = ExportToTFServingSpec(encoding_dir=encoding_dir,
                                     checkpoint_dir=checkpoint_dir,
                                     export_dir=export_dir,
                                     train_config=train_config)

        super(ExportToTFServing, self).__init__(spec=spec)
