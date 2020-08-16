import logging
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2

from tfx_gpt2.gpt_2 import model, sample, encoder
from tfx_gpt2.gpt_2.load_dataset import load_dataset, Sampler
from tfx_gpt2.gpt_2.accumulate import AccumulatingOptimizer
from tfx_gpt2.gpt_2 import memory_saving_gradients

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


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32)
        return tf.where(mask, noise, context)
    else:
        return context


def train_gpt2(dataset_path,
               model_path,
               model_name,
               train_config,
               combine,
               encoding,
               checkpoint_dir,
               sample_dir):
    enc = encoder.get_encoder(model_path)
    hparams = model.default_hparams()
    with open(os.path.join(model_path, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if train_config["sample_length"] > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    if model_name == '345M':
        train_config["memory_saving_gradients"] = True
        if train_config["optimizer"] == 'adam':
            train_config["only_train_transformer_layers"] = True

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [train_config["batch_size"], None])
        context_in = randomize(context, hparams, train_config["noise"])
        output = model.model(hparams=hparams, X=context_in)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=train_config["sample_length"],
            context=context,
            batch_size=train_config["batch_size"],
            temperature=1.0,
            top_k=train_config["top_k"],
            top_p=train_config["top_p"])

        all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in all_vars if '/h' in v.name] if train_config[
            "only_train_transformer_layers"] else all_vars

        if train_config["optimizer"] == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=train_config["learning_rate"])
        elif train_config["optimizer"] == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=train_config["learning_rate"])
        else:
            logging.info('Bad optimizer only adam or sgd available:', train_config["optimizer"])
            return 0

        if train_config["accumulate_gradients"] > 1:
            if train_config["memory_saving_gradients"]:
                logging.info("Memory saving gradients are not implemented for gradient accumulation yet.")
                return 0

            opt = AccumulatingOptimizer(
                opt=opt,
                var_list=train_vars)
            opt_reset = opt.reset()
            opt_compute = opt.compute_gradients(loss)
            opt_apply = opt.apply_gradients()
            summary_loss = tf.summary.scalar('loss', opt_apply)
        else:
            if train_config["memory_saving_gradients"]:
                opt_grads = memory_saving_gradients.gradients(loss, train_vars)
            else:
                opt_grads = tf.gradients(loss, train_vars)
            opt_grads = list(zip(opt_grads, train_vars))
            opt_apply = opt.apply_gradients(opt_grads)
            summary_loss = tf.summary.scalar('loss', loss)

        summary_lr = tf.summary.scalar('learning_rate', train_config["learning_rate"])
        summaries = tf.summary.merge([summary_lr, summary_loss])

        summary_log = tf.summary.FileWriter(
            os.path.join(checkpoint_dir))

        saver = tf.train.Saver(
            var_list=all_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())

        # Get fresh GPT weights if new run.
        ckpt = tf.train.latest_checkpoint(
            os.path.join(model_path))
        logging.info('Loading checkpoint', ckpt)
        saver.restore(sess, ckpt)

        logging.info('Loading dataset...')
        chunks = load_dataset(enc, dataset_path, combine, encoding=encoding)
        data_sampler = Sampler(chunks)
        logging.info('dataset has', data_sampler.total_size, 'tokens')
        logging.info('Training...')

        counter = 1
        counter_path = os.path.join(checkpoint_dir, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            logging.info('Saving {} model-{}'.format(checkpoint_dir, counter))
            saver.save(
                sess,
                os.path.join(checkpoint_dir, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')

        def generate_samples():
            logging.info('Generating samples...')
            context_tokens = data_sampler.sample(1)
            all_text = []
            index = 0
            while index < train_config["sample_num"]:
                out = sess.run(
                    tf_sample,
                    feed_dict={context: train_config["batch_size"] * [context_tokens]})
                for i in range(min(train_config["sample_num"] - index, train_config["batch_size"])):
                    text = enc.decode(out[i])
                    text = '======== SAMPLE {} ========\n{}\n'.format(
                        index + 1, text)
                    all_text.append(text)
                    index += 1
            with open(os.path.join(sample_dir, 'samples-{}').format(counter), 'w', encoding=encoding) as fp:
                fp.write('\n'.join(all_text))

        def sample_batch():
            return [data_sampler.sample(1024) for _ in range(train_config["batch_size"])]

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        while counter < train_config["num_iterations"]:
            if counter % train_config["save_every"] == 0:
                save()
            if counter % train_config["sample_every"] == 0:
                generate_samples()

            if train_config["accumulate_gradients"] > 1:
                sess.run(opt_reset)
                for _ in range(train_config["accumulate_gradients"]):
                    sess.run(
                        opt_compute, feed_dict={context: sample_batch()})
                (v_loss, v_summary) = sess.run((opt_apply, summaries))
            else:
                (_, v_loss, v_summary) = sess.run(
                    (opt_apply, loss, summaries),
                    feed_dict={context: sample_batch()})

            summary_log.add_summary(v_summary, counter)

            avg_loss = (avg_loss[0] * 0.99 + v_loss,
                        avg_loss[1] * 0.99 + 1.0)

            logging.info('[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'.format(
                counter=counter,
                time=time.time() - start_time,
                loss=v_loss,
                avg=avg_loss[0] / avg_loss[1]))

            counter += 1


class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        model_name = exec_properties["model_name"]
        combine = exec_properties["combine"]
        encoding = exec_properties["encoding"]
        train_config = exec_properties["train_config"]

        dataset_path = get_single_uri(input_dict["dataset_path"])
        model_path = get_single_uri(input_dict["model_path"])

        checkpoint_dir = get_single_uri(output_dict["checkpoint_dir"])
        sample_dir = get_single_uri(output_dict["sample_dir"])

        train_gpt2(dataset_path=dataset_path,
                   model_path=model_path,
                   model_name=model_name,
                   train_config=train_config,
                   combine=combine,
                   encoding=encoding,
                   checkpoint_dir=checkpoint_dir,
                   sample_dir=sample_dir)


class TrainGPT2Spec(types.ComponentSpec):
    PARAMETERS = {
        'model_name': ExecutionParameter(type=Text),
        'combine': ExecutionParameter(type=int),
        'encoding': ExecutionParameter(type=Text),
        'train_config': ExecutionParameter(type=Dict),

    }

    INPUTS = {
        'dataset_path': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'model_path': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }

    OUTPUTS = {
        'checkpoint_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'sample_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),

    }


class TrainGPT2(base_component.BaseComponent):
    SPEC_CLASS = TrainGPT2Spec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 dataset_path: types.Channel,
                 model_path: types.Channel,
                 model_name: Text,
                 train_config: Dict,
                 combine: int = 50000,
                 encoding: Text = 'utf-8'):
        checkpoint_dir = external_input("TrainGPT2")
        sample_dir = external_input("TrainGPT2")

        spec = TrainGPT2Spec(dataset_path=dataset_path,
                             model_path=model_path,
                             model_name=model_name,
                             train_config=train_config,
                             combine=combine,
                             encoding=encoding,
                             checkpoint_dir=checkpoint_dir,
                             sample_dir=sample_dir)

        super(TrainGPT2, self).__init__(spec=spec)
