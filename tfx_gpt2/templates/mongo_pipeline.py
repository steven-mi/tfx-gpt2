import os

from tfx.orchestration import metadata
from tfx.orchestration import pipeline

from tfx_gpt2.components.create_dataset import CreateDataset
from tfx_gpt2.components.download_model import DownloadPretrainedModel
from tfx_gpt2.components.export_for_tfserving import ExportToTFServing
from tfx_gpt2.components.mlflow_import import MLFlowImport
from tfx_gpt2.components.mongo_export import MongoExport
from tfx_gpt2.components.train_gpt2 import TrainGPT2
from tfx_gpt2.templates import default_train_config


def create_pipeline(pipeline_name, pipeline_root, model_name, train_config, mlflow_tracking_url, mongo_ip,
                    mongo_colnames, mongo_port="27017", encoding='utf-8', combine=50000, enable_cache=False):
    for key, value in train_config.items():
        default_train_config[key] = value

    mongo_export = MongoExport(colnames=mongo_colnames,
                               ip=mongo_ip,
                               port=mongo_port)

    pretrained_model = DownloadPretrainedModel(model_name=model_name)

    create_dataset = CreateDataset(text_path=mongo_export.outputs["output_dir"],
                                   model_path=pretrained_model.outputs["model_path"],
                                   encoding=encoding,
                                   combine=combine)

    train_gpt2 = TrainGPT2(dataset_path=create_dataset.outputs["dataset_path"],
                           model_path=pretrained_model.outputs["model_path"],
                           model_name=model_name,
                           train_config=train_config,
                           combine=combine,
                           encoding=encoding)
    export_tfserving = ExportToTFServing(model_path=pretrained_model.outputs["model_path"],
                                         checkpoint_dir=train_gpt2.outputs["checkpoint_dir"],
                                         train_config=train_config)

    mlflow_import = MLFlowImport(model_name=model_name,
                                 mlflow_tracking_url=mlflow_tracking_url,
                                 artifact_dir=train_gpt2.outputs["sample_dir"],
                                 hyperparameter_dir=train_gpt2.outputs["hyperparameter_dir"],
                                 metric_dir=train_gpt2.outputs["metric_dir"],
                                 model_dir=train_gpt2.outputs["export_dir"])

    pipeline_root = os.path.join(pipeline_root, 'pipelines', pipeline_name)
    metadata_path = os.path.join(pipeline_root, 'metadata', pipeline_name,
                                 'metadata.db')

    tfx_pipeline = pipeline.Pipeline(pipeline_name=pipeline_name,
                                     pipeline_root=pipeline_root,
                                     components=[create_dataset,
                                                 pretrained_model,
                                                 train_gpt2,
                                                 export_tfserving,
                                                 mlflow_import],
                                     enable_cache=enable_cache,
                                     metadata_connection_config=metadata.sqlite_metadata_connection_config(
                                         metadata_path))
    return tfx_pipeline
