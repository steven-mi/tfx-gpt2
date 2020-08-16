# TODO

from tfx_gpt2.components.create_dataset import CreateDataset
from tfx_gpt2.components.download_model import DownloadPretrainedModel
from tfx_gpt2.components.train_gpt2 import TrainGPT2


def create_components(model_name, text_path, train_config, encoding='utf-8', combine=50000):
    pretrained_model = DownloadPretrainedModel(model_name=model_name)

    create_dataset = CreateDataset(text_path=text_path,
                                   model_path=pretrained_model.outputs["model_path"],
                                   encoding=encoding,
                                   combine=combine)

    train_gpt2 = TrainGPT2(dataset_path=create_dataset.outputs["dataset_path"],
                           model_path=pretrained_model.outputs["model_path"],
                           model_name=model_name,
                           train_config=train_config,
                           combine=combine,
                           encoding=encoding)

    return [create_dataset,
            pretrained_model,
            train_gpt2]
