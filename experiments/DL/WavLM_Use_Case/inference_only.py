#!/usr/bin/env python3
import os
import sys
import csv
import json
import torch
import numpy as np
from enum import Enum, auto
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader
#from sklearn.metrics import f1_score
from transformers import WavLMModel
import speechbrain as sb
import pickle

# Prevent SpeechBrain from crashing when it tries to use CUDA on CPU-only machine
# torch.cuda.is_available = lambda: False
# torch.cuda.set_device = lambda x: None

FILEPATH = os.environ['FILEPATH']
OUTPUT_PATH = os.environ['OUTPUT_PATH']
TOOL = os.environ['TOOL']
MODE = os.environ['MODE']

class Stage(Enum):
    TEST = auto()

class EmoIdBrain(sb.Brain):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        scratch_dir = os.path.join(FILEPATH, 'wavlm-large')
        self.wavlm_model = WavLMModel.from_pretrained(scratch_dir)
        self.wavlm_model.eval().to(self.device)
        self.predictions, self.true_labels, self.genders = [], [], []

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        with torch.no_grad():
            wavs = torch.nn.functional.layer_norm(wavs.float(), wavs.shape)
            if len(wavs.shape) == 3:
                wavs = wavs.squeeze(1)
            feats = self.wavlm_model(wavs).last_hidden_state

        embeddings = self.modules.embedding_model(feats, lens)
        if TOOL != 'ieee':
            pickle.dump(embeddings, open(f"{OUTPUT_PATH}/embeddings_{TOOL}_{MODE}_{os.environ['ITER']}_{os.environ.get('SUBJECT_ID')}.pkl", 'wb'))
        else:
            pickle.dump(embeddings, open(f"{OUTPUT_PATH}/embeddings_{TOOL}_{os.environ['ITER']}_{os.environ.get('SUBJECT_ID')}.pkl", 'wb'))

        output = self.modules.classifier(embeddings)
        if TOOL != 'ieee':
            pickle.dump(output, open(f"{OUTPUT_PATH}/output_{TOOL}_{MODE}_{os.environ['ITER']}_{os.environ.get('SUBJECT_ID')}.pkl", 'wb'))
        else:
            pickle.dump(output, open(f"{OUTPUT_PATH}/output_{TOOL}_{os.environ['ITER']}_{os.environ.get('SUBJECT_ID')}.pkl", 'wb'))

        return torch.nn.functional.log_softmax(output, dim=-1)

    def output_predictions(self, test_set, output_path, test_loader_kwargs={}):
        if not isinstance(test_set, DataLoader):
            test_set = self.make_dataloader(test_set, Stage.TEST, **test_loader_kwargs)

        with open(output_path, "w", newline="") as f:
            csv.writer(f).writerow(["id", "prediction", "true_value", "gender"])

        self.modules.eval()
        with torch.no_grad():
            for batch in tqdm(test_set):
                ids = batch.id
                true_vals = batch.label_encoded.data.squeeze(1).tolist()
                genders = batch.gender
                # preds = torch.argmax(self.compute_forward(batch, Stage.TEST), dim=-1).squeeze(1).tolist()
                preds = self.compute_forward(batch, Stage.TEST)
                preds = torch.argmax(preds, dim=-1)
                preds = preds.squeeze(1)
                preds = preds.tolist()

                with open(output_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for i, p, t, g in zip(ids, preds, true_vals, genders):
                        writer.writerow([i, p, t, g])
                        print(f"ID: {i}, Prediction: {p}, True: {t}, Gender: {g}")

def dataio_prep(hparams):
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        wav = wav.replace("__DATA_PATH__", hparams["data_folder"])
        try:
            return sb.dataio.dataio.read_audio(wav)
        except:
            return torch.zeros(16000)

    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    @sb.utils.data_pipeline.takes("label", "gender")
    @sb.utils.data_pipeline.provides("label", "label_encoded", "gender")
    def label_pipeline(label, gender):
        yield label
        yield label_encoder.encode_label_torch(label)
        yield gender

    # Load full dataset first (unfiltered)
    filtered_dataset = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": hparams["data_folder"]},
    )

    # Load label encoder from all data (important)
    label_encoder.load_or_create(
        path=os.path.join(hparams["save_folder"], "label_encoder.txt"),
        from_didatasets=[filtered_dataset],
        output_key="label"
    )

    subject_id = os.environ.get("SUBJECT_ID")
    if subject_id is not None:
        filtered_data = {
        # k: v for k, v in dataset.data.items() if v["wav"] == f"__DATA_PATH__/{os.environ['SUBJECT_ID']}.wav"
            k: v for k, v in filtered_dataset.data.items() if k == subject_id
        }
        
        dataset = sb.dataio.dataset.DynamicItemDataset(filtered_data)

    # Apply dynamic items after filtering
    dataset.add_dynamic_item(audio_pipeline)
    dataset.add_dynamic_item(label_pipeline)
    dataset.set_output_keys(["id", "sig", "label_encoded", "gender"])

    return dataset

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    test_set = dataio_prep(hparams)

    brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=None,
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Load the best checkpoint
    brain.checkpointer.recover_if_possible()

    # Run inference
    output_csv = os.path.join(hparams["output_folder"], "predictions.csv")
    brain.output_predictions(test_set, output_csv, test_loader_kwargs=hparams["dataloader_options"])
