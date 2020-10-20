import pickle
import logging
import json
import os
import tensorflow as tf
import time
from tensorflow.core.protobuf import rewriter_config_pb2

from tfx_gpt2.core import model, sample, encoder
from tfx_gpt2.core.load_dataset import load_dataset, Sampler
from tfx_gpt2.core.accumulate import AccumulatingOptimizer
from tfx_gpt2.core import memory_saving_gradients

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


def train_gpt2(dataset_dir, checkpoint_dir, encoding_dir,
               model_name, train_config, encoding,
               trained_checkpoint_dir, sample_dir, tensorboard_dir,
               end_token):
    enc = encoder.get_encoder(encoding_dir)
    hparams = model.default_hparams()
    with open(os.path.join(encoding_dir, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if train_config["sample_length"] > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

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
            top_p=train_config["top_p"],
            start_token=end_token)

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
            os.path.join(tensorboard_dir))

        saver = tf.train.Saver(
            var_list=train_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())

        try:
            ckpt = tf.train.latest_checkpoint(
                os.path.join(checkpoint_dir))
            logging.info('Loading checkpoint', ckpt)
            saver.restore(sess, ckpt)
        except:
            logging.info("Loading checkpoint failed - training from scratch")

        logging.info('Loading dataset...')
        chunks = load_dataset(enc, dataset_dir, encoding=encoding, end_token=end_token)
        logging.info('Loading dataset to sampler')
        data_sampler = Sampler(chunks)
        logging.info('Sample 1: {}'.format(data_sampler.sample(1)))
        logging.info('Sample 1: {}'.format(data_sampler.sample(1)))
        logging.info('dataset has {} tokens'.format(data_sampler.total_size))

        logging.info('Training {}...'.format(model_name))
        counter = 1
        counter_path = os.path.join(trained_checkpoint_dir, 'counter')
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, 'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            logging.info('Saving {} model-{}'.format(trained_checkpoint_dir, counter))
            saver.save(
                sess,
                os.path.join(trained_checkpoint_dir, 'model'),
                global_step=counter)
            with open(counter_path, 'w') as fp:
                fp.write(str(counter) + '\n')

        def generate_samples():
            logging.info('Generating samples...')
            context_tokens = data_sampler.sample(train_config["batch_size"])
            all_text = []
            index = 0
            while index < train_config["sample_num"]:
                out = sess.run(
                    tf_sample,
                    feed_dict={context: train_config["batch_size"] * [context_tokens]})
                for i in range(min(train_config["sample_num"] - index, train_config["batch_size"])):
                    text = 'Input: {} ======== SAMPLE {} ========\n{}\n'.format(enc.decode(context_tokens),
                                                                                index + 1, enc.decode(out[i]))
                    all_text.append(text)
                    index += 1
            with open(os.path.join(sample_dir, 'samples-{}').format(counter), 'w', encoding=encoding) as fp:
                fp.write('\n'.join(all_text))

        def sample_batch():
            return [data_sampler.sample(train_config["sample_length"]) for _ in range(train_config["batch_size"])]

        avg_loss = (0.0, 0.0)
        start_time = time.time()
        while counter < train_config["num_iterations"]:
            if counter % train_config["save_every"] == 0:
                save()
            if counter % train_config["sample_every"] == 0:
                generate_samples()
                logging.info(counter)

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
    return train_config, {"loss": v_loss,
                          "avg loss": avg_loss[0] / avg_loss[1]}


class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        model_name = exec_properties["model_name"]
        encoding = exec_properties["encoding"]
        train_config = exec_properties["train_config"]
        end_token = exec_properties["end_token"]
        dataset_dir = get_single_uri(input_dict["dataset_dir"])
        checkpoint_dir = get_single_uri(input_dict["checkpoint_dir"])
        encoding_dir = get_single_uri(input_dict["encoding_dir"])

        trained_checkpoint_dir = get_single_uri(output_dict["trained_checkpoint_dir"])
        sample_dir = get_single_uri(output_dict["sample_dir"])
        tensorboard_dir = get_single_uri(output_dict["tensorboard_dir"])
        hyperparameter_dir = get_single_uri(output_dict["hyperparameter_dir"])
        metric_dir = get_single_uri(output_dict["metric_dir"])
        train_config, metrics = train_gpt2(dataset_dir=dataset_dir,
                                           checkpoint_dir=checkpoint_dir,
                                           encoding_dir=encoding_dir,
                                           model_name=model_name,
                                           train_config=train_config,
                                           encoding=encoding,
                                           trained_checkpoint_dir=trained_checkpoint_dir,
                                           sample_dir=sample_dir,
                                           tensorboard_dir=tensorboard_dir,
                                           end_token=end_token)

        with open(os.path.join(hyperparameter_dir, 'hyperparameter.pickle'), 'wb') as handle:
            pickle.dump(train_config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(metric_dir, 'metric.pickle'), 'wb') as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)


class TrainGPT2Spec(types.ComponentSpec):
    PARAMETERS = {
        'model_name': ExecutionParameter(type=Text),
        'encoding': ExecutionParameter(type=Text),
        'end_token': ExecutionParameter(type=Text),
        'train_config': ExecutionParameter(type=Dict),
    }

    INPUTS = {
        'dataset_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'checkpoint_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'encoding_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }

    OUTPUTS = {
        'trained_checkpoint_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'tensorboard_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'sample_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'hyperparameter_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
        'metric_dir': ChannelParameter(type=standard_artifacts.ExternalArtifact),
    }


class TrainGPT2(base_component.BaseComponent):
    SPEC_CLASS = TrainGPT2Spec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(Executor)

    def __init__(self,
                 dataset_dir: types.Channel,
                 checkpoint_dir: types.Channel,
                 encoding_dir: types.Channel,
                 model_name: Text,
                 train_config: Dict,
                 encoding: Text = 'utf-8',
                 end_token: Text = ""):
        trained_checkpoint_dir = external_input("TrainGPT2")
        sample_dir = external_input("TrainGPT2")
        tensorboard_dir = external_input("TrainGPT2")
        hyperparameter_dir = external_input("TrainGPT2")
        metric_dir = external_input("TrainGPT2")

        spec = TrainGPT2Spec(dataset_dir=dataset_dir,
                             checkpoint_dir=checkpoint_dir,
                             encoding_dir=encoding_dir,
                             model_name=model_name,
                             train_config=train_config,
                             encoding=encoding,
                             trained_checkpoint_dir=trained_checkpoint_dir,
                             sample_dir=sample_dir,
                             hyperparameter_dir=hyperparameter_dir,
                             metric_dir=metric_dir,
                             tensorboard_dir=tensorboard_dir,
                             end_token=end_token)

        super(TrainGPT2, self).__init__(spec=spec)
