from typing import Sequence
import os
import datasets
from datasets import DownloadManager
DESCRICAO = "Brasilian Portuguese NER dataset"

def get_sentencas(corpus):
        sentenca = []
        for line in corpus:
            if line == '\n':
                if sentenca:
                    yield sentenca
                    sentenca = []
            else:
                
                sentenca.append(line.strip('\n').split('\t'))
        if sentenca:
            yield sentenca
class CustomDataset(datasets.GeneratorBasedBuilder):
    
    def _info(self):
        features = datasets.Features({
            "id": datasets.Value("string"),
            "tokens": datasets.Sequence(datasets.Value("string")),
            "ner_tags": datasets.Sequence(
                datasets.features.ClassLabel(
                    names=[
                        'O',
                        'TEMPO',
                        'PESSOA',
                        'ORGANIZACAO',
                        'LOCAL',
                ])
            ),
        })

        return datasets.DatasetInfo(
            description=DESCRICAO,

            features=features
        )
    
    def _split_generators(self, dl_manager: DownloadManager):
        # data_file = "corpus_First_HAREM.txt"
        _URL = "/home/rthr/Documentos/2022-2/IC/Base de Dados/Paramopama/divisions/iterative/"
        _URLS = {
        # "train": _URL + "train_data.colnn",
        # "validation": _URL + "val_data.colnn",
        # "test": _URL + "test_data.colnn",

        "particao_0": _URL + "division_0",
        "particao_1": _URL + "division_1",
        "particao_2": _URL + "division_2",
        "particao_3": _URL + "division_3",
        "particao_4": _URL + "division_4",
        "particao_5": _URL + "division_5",
        "particao_6": _URL + "division_6",
        "particao_7": _URL + "division_7",
        "particao_8": _URL + "division_8",
        "particao_9": _URL + "division_9",
        }
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return[
            # datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            # datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["validation"]}),
            # datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),

            datasets.SplitGenerator(name='particao_0', gen_kwargs={"filepath": downloaded_files["particao_0"]}),
            datasets.SplitGenerator(name='particao_1', gen_kwargs={"filepath": downloaded_files["particao_1"]}),
            datasets.SplitGenerator(name='particao_2', gen_kwargs={"filepath": downloaded_files["particao_2"]}),
            datasets.SplitGenerator(name='particao_3', gen_kwargs={"filepath": downloaded_files["particao_3"]}),
            datasets.SplitGenerator(name='particao_4', gen_kwargs={"filepath": downloaded_files["particao_4"]}),
            datasets.SplitGenerator(name='particao_5', gen_kwargs={"filepath": downloaded_files["particao_5"]}),
            datasets.SplitGenerator(name='particao_6', gen_kwargs={"filepath": downloaded_files["particao_6"]}),
            datasets.SplitGenerator(name='particao_7', gen_kwargs={"filepath": downloaded_files["particao_7"]}),
            datasets.SplitGenerator(name='particao_8', gen_kwargs={"filepath": downloaded_files["particao_8"]}),
            datasets.SplitGenerator(name='particao_9', gen_kwargs={"filepath": downloaded_files["particao_9"]}),                        
            ]
    

    
    def _generate_examples(self, filepath):

        
        sentence = 0
        tokens = []
        ner_tags = []

        with open(filepath, "r") as corpus:
            data = list(get_sentencas(corpus))
        
        for idx, text in enumerate(data):
            
            if tokens != []:
                yield sentence, {
                    "id": sentence,
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
            sentence = idx
            tokens = []
            ner_tags = []

            for token, ner_tag in text:
                tokens.append(token)
                ner_tags.append(ner_tag)

        yield sentence, {
                    "id": sentence,
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                    }