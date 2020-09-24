import os

from tfx.orchestration import metadata
from tfx.orchestration import pipeline

from tfx_gpt2.components.create_encoded_dataset import CreateEncodedDataset
from tfx_gpt2.components.create_encoding import CreateEncoding
from tfx_gpt2.components.create_merged_text import CreateMergedText
from tfx_gpt2.components.download_pretrained_model import DownloadPretrainedModel
from tfx_gpt2.components.export_for_tfserving import ExportToTFServing
from tfx_gpt2.components.mlflow_import import MLFlowImport
from tfx_gpt2.components.mongo_export import MongoExport
from tfx_gpt2.components.train_gpt2 import TrainGPT2
from tfx_gpt2.templates import default_train_config


def create_pipeline(pipeline_name, pipeline_root, model_name, train_config, mlflow_tracking_url,
                    mongo_ip, mongo_colnames, mongo_port="27017",
                    encoding='utf-8', text_token_size=50000, enable_cache=False, end_token="<|endoftext|>"):
    for key, value in train_config.items():
        default_train_config[key] = value

    mongo_export = MongoExport(colnames=mongo_colnames,
                               ip=mongo_ip,
                               port=mongo_port,
                               end_token=end_token)

    pretrained_model = DownloadPretrainedModel(model_name=model_name)

    create_encoding = CreateEncoding(encoding=encoding,
                                     model_dir=pretrained_model.outputs["model_dir"],
                                     merged_text_dir=mongo_export.outputs["merged_text_dir"],
                                     text_token_size=text_token_size,
                                     end_token=end_token)

    create_dataset = CreateEncodedDataset(merged_text_dir=mongo_export.outputs["merged_text_dir"],
                                          encoding_dir=create_encoding.outputs["encoding_dir"],
                                          encoding=encoding)

    train_gpt2 = TrainGPT2(dataset_dir=create_dataset.outputs["dataset_dir"],
                           checkpoint_dir=pretrained_model.outputs["model_dir"],
                           encoding_dir=create_encoding.outputs["encoding_dir"],
                           model_name=model_name,
                           train_config=train_config,
                           encoding=encoding)

    export_tfserving = ExportToTFServing(encoding_dir=create_encoding.outputs["encoding_dir"],
                                         checkpoint_dir=train_gpt2.outputs["trained_checkpoint_dir"],
                                         train_config=train_config)

    mlflow_import = MLFlowImport(model_name=model_name,
                                 mlflow_tracking_url=mlflow_tracking_url,
                                 artifact_dir=train_gpt2.outputs["sample_dir"],
                                 hyperparameter_dir=train_gpt2.outputs["hyperparameter_dir"],
                                 metric_dir=train_gpt2.outputs["metric_dir"],
                                 model_dir=export_tfserving.outputs["export_dir"])

    pipeline_path = os.path.join(pipeline_root, 'pipelines', pipeline_name)
    metadata_path = os.path.join(pipeline_root, 'metadata', pipeline_name,
                                 'metadata.db')

    tfx_pipeline = pipeline.Pipeline(pipeline_name=pipeline_name,
                                     pipeline_root=pipeline_path,
                                     components=[mongo_export,
                                                 pretrained_model,
                                                 create_encoding,
                                                 create_dataset,
                                                 train_gpt2,
                                                 export_tfserving,
                                                 mlflow_import],
                                     enable_cache=enable_cache,
                                     metadata_connection_config=metadata.sqlite_metadata_connection_config(
                                         metadata_path))
    return tfx_pipeline
