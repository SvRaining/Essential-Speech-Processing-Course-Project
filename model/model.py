import gc
import json
import os
import re
import shutil
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import spacy
import textstat
import torch
import wandb
from datasets import Dataset, disable_progress_bar
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import PredictionOutput


class DataPaths:
    TRAIN_PROMPTS = "train_prompts.csv"
    TEST_PROMPTS = "test_prompts.csv"
    TRAIN_SUMMARIES = "train_summaries.csv"
    TEST_SUMMARIES = "test_summaries.csv"


class WandbMetrics:
    DEBERTA_METRIC_1 = "metric-deberta-1"
    DEBERTA_METRIC_2 = "metric-deberta-2"
    DEBERTA_METRIC_3 = "metric-deberta-3"
    LGBM_METRIC_1 = "metric-lgbm-1"
    LGBM_METRIC_2 = "metric-lgbm-2"
    LGBM_METRIC_3 = "metric-lgbm-3"


FOLD_MAP = {
    "a1b2c3": 0,
    "b2c3d4": 1,
    "c3d4e5": 2,
    "d4e5f6": 3,
}

DATA_ENV_VAR = "DATA_ENV_VAR"
DRY_RUN_FLAG = "DRY_RUN_FLAG"
DEFAULT_DATA_PATH = "/data/resources/"
TEMP_PATH = "temp_storage"
MODEL_PATH = "saved_models"
CLASS_LABELS = ["item_content", "item_wording"]
PRED_LABELS = [f"prediction_{label}" for label in CLASS_LABELS]
SAVE_TOP_MODELS = 3
OLD_KEYS = [
    "legacy_add_text_short",
    "data_directory",
    "processors_count",
]


def fetch_tags():
    if any(key.startswith("DATA_ENV") for key in os.environ):
        return ["data_environment"]
    else:
        return ["local_environment"]


def general_configuration(set_cublas=True, online=True):
    warnings.filterwarnings("default")
    os.environ["TOKENIZER_PARALLEL"] = "disabled"
    tqdm.set_postfix()  # update
    turn_off_progress_bar()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if set_cublas:
        os.environ["CUBLAS_SPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    try:
        if any(key.startswith("DATA_ENV") for key in os.environ):
            if online:
                from data_secrets import SecretsHandler
                secrets = SecretsHandler()
                secret_key = secrets.fetch_secret("wandb_auth")
                wandb.authenticate(key=secret_key)
        else:
            if online:
                wandb.authenticate()
            os.environ[DATA_ENV_VAR] = "data_resources/"
    except:
        print("Failed to authenticate with WandB")
        exit(1)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = os.environ.get(DATA_ENV_VAR, DEFAULT_DATA_PATH)
    if not Path(data_dir).is_dir():
        raise FileNotFoundError(f"Cannot find directory {data_dir}")

    sub_dir = Path(data_dir) / "student_summaries_data"

    train_prompts = pd.read_csv(sub_dir / DataPaths.TRAIN_PROMPTS)
    test_prompts = pd.read_csv(sub_dir / DataPaths.TEST_PROMPTS)
    train_summaries = pd.read_csv(sub_dir / DataPaths.TRAIN_SUMMARIES)
    test_summaries = pd.read_csv(sub_dir / DataPaths.TEST_SUMMARIES)

    return train_prompts, test_prompts, train_summaries, test_summaries


@dataclass
class Settings:
    model_path: str = "model/deberta-v3-default"
    max_tokens: int = 512
    include_question: bool = False
    include_text: bool = False
    dropout_hidden: float = 0.00
    dropout_attention: float = 0.00
    learning_rate: float = 1e-05
    decay: float = 0.0
    batch_size: int = 6
    training_cycles: int = 3
    randomness_seed: int = 42
    report_dest: str = "wandb"
    eval_interval: int = 50
    early_stop: int = 15
    warmup_cycles: int = 200
    use_16bit: Optional[bool] = None
    pseudo_label: bool = False
    frozen_layers: int = 0
    accumulate_steps: int = 1
    grad_checkpoint: bool = False

    def __post_init__(self):
        if self.use_16bit is None:
            if "large" in self.model_path:
                self.use_16bit = True
            else:
                self.use_16bit = False


def process_text(data, text_processor, settings: Settings, labeled: bool = True):
    separator = f" {text_processor.sep_token} "
    elements = []
    elements.append("text_info")
    if settings.include_question:
        elements.append("query_question")
    if settings.include_text:
        elements.append("query_text")

    tokenized_data = text_processor(
        separator.join([data[element] for element in elements]).strip(),
        padding=False,
        truncation=True,
        max_length=settings.max_tokens,
    )

    processed = {**tokenized_data}

    if labeled:
        processed["labels"] = [data[label] for label in CLASS_LABELS]

    return processed


def compute_accuracy(trainer_data: Tuple) -> Dict:
    predictions, actuals = trainer_data
    col_error = np.sqrt(np.mean((predictions - actuals) ** 2, axis=0))
    mean_error = np.mean(col_error)

    output = {f"{label}_error": col_error[i]
              for i, label in enumerate(CLASS_LABELS)}
    output["mean_error"] = mean_error

    return output


def disable_training(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


class BaseModule:
    """Base class for one segment"""

    def __init__(self, model_ref: str, buffer_zone: str = TEMP_AREA, save_loc: str = MODEL_SAVE_AREA):
        self.model_ref = model_ref
        self.buffer_zone = buffer_zone
        self.save_loc = save_loc

    def train_segment(self, params: Parameters, segment: int) -> ResultAnalysis:
        model_params = AutoParams.from_pretrained(self.model_ref)
        txt_parser = AutoParser.from_pretrained(self.model_ref)
        data_helper = DataHelperWithLimit(txt_parser)
        model_params.update({
            "label_count": 2,
            "task_type": "value_prediction",
        })

        if params.report_to_service == "webtool":
            activity = webtool.start(
                project=WebtoolProjects.PROJECT_ABC,
                params=params.to_dict(),
                labels=get_webtool_labels() + [f"segment_{segment}"],
            )
        set_random_seed(params.rand_seed)
        model_params.update({
            "drop_prob": params.drop_prob,
            "attention_drop_prob": params.attention_drop_prob,
            "params_cfg": params.to_dict(),
        })

        training_params = TrainingParams(
            dest=TEMP_AREA,
            pick_best=True,
            rate=params.learn_rate,
            batch_train=params.train_batch,
            batch_eval=1,
            total_epochs=params.epoch_count,
            decay_rate=params.decay_rate,
            report_dest=params.report_to_service,
            better_is_higher=False,
            save_way="intervals",
            eval_way="intervals",
            eval_intervals=params.check_every,
            save_intervals=params.check_every,
            best_metric="m_score",
            max_saves=1,
            log_intervals=params.check_every,
            rand_seed=params.rand_seed,
            use_16_bit=params.use_16_bit,
            warm_intervals=params.warm_intervals,
            accumulate_steps=params.accumulate_steps,
            check_gradients=params.check_gradients,
        )
        model_instance = AutoModelForClassification.from_pretrained(
            params.model_ref, params=model_params
        )

        if params.layer_lock > 0:
            disable_parameters(model_instance.deberta.embeddings)
            for i in range(params.layer_lock):
                disable_parameters(model_instance.deberta.encoder.layer[i])

        if params.pseudo_mode:
            pseudo_params = deepcopy(params)
            pseudo_params.prompt_text = False
            pseudo_params.prompt_question = False
            dataset_path = os.environ.get(DATA_ENV_VAR, DEFAULT_DATA_PATH)
            data_ref = Path(
                dataset_path) / "feedback-prize-english-language-learning" / "train.csv"
            assert data_ref.exists()
            data_content = pd.read_csv(data_ref)

            pseudo_data_path = "sandbox/datasets/fbp3_pseudolabelling/label_data.csv"
            pseudo_data_ref = Path(pseudo_data_path)
            assert pseudo_data_ref.exists()
            pseudo_data_content = pd.read_csv(pseudo_data_ref)

            training_data = pd.merge(
                data_content, pseudo_data_content, left_on="text_id", right_on="student_id")
            training_data = training_data.rename(columns={"full_text": "text"})

            placeholder = -1000
            split_count = 10
            training_data["segment"] = placeholder
            segment_split = SegmentSplitter(
                n_splits=split_count, shuffle=True, random_seed=pseudo_params.rand_seed)
            for n, (train_idx, validate_idx) in enumerate(segment_split.split(training_data, training_data[RESULT_LABELS])):
                training_data.loc[validate_idx, "segment"] = int(n)
            assert not (training_data["segment"] == placeholder).any()

            pseudo_training_data = Dataset.from_data_frame(
                training_data[training_data["segment"] != segment])
            pseudo_validation_data = Dataset.from_data_frame(
                training_data[training_data["segment"] == segment])
            pseudo_training_data = pseudo_training_data.map(parser_function, batched=False, fn_params={
                                                            "parser": txt_parser, "params": pseudo_params})
            pseudo_validation_data = pseudo_validation_data.map(
                parser_function, batched=False, fn_params={"parser": txt_parser, "params": pseudo_params})

            pseudo_trainer_instance = Trainer(
                model=model_instance,
                params=training_params,
                train_data=pseudo_training_data,
                eval_data=pseudo_validation_data,
                data_helper=data_helper,
                parser=txt_parser,
                metrics_func=calculate_m_score,
                callbacks=[StopEarly(patience=5)],
            )
            pseudo_trainer_instance.train()

        # Main training
        primary_train, primary_test, summary_train, summary_test = fetch_data()
        merge_data = primary_train.merge(summary_train, on="prompt_id")
        assert set(ID_MAP.keys()) == set(merge_data["prompt_id"].unique())
        merge_data["segment"] = merge_data["prompt_id"].map(ID_MAP)
        training_data = Dataset.from_data_frame(
            merge_data[merge_data["segment"] != segment])
        validation_data = Dataset.from_data_frame(
            merge_data[merge_data["segment"] == segment])

        training_data = training_data.map(parser_function, batched=False, fn_params={
                                          "parser": txt_parser, "params": params})
        validation_data = validation_data.map(parser_function, batched=False, fn_params={
                                              "parser": txt_parser, "params": params})

        trainer_instance = Trainer(
            model=model_instance,
            params=training_params,
            train_data=training_data,
            eval_data=validation_data,
            data_helper=data_helper,
            parser=txt_parser,
            metrics_func=calculate_m_score,
            callbacks=[StopEarly(patience=5)],
        )
        trainer_instance.train()

        result_info = trainer_instance.evaluate()

        if params.report_to_service == "webtool":
            webtool.end(project=WebtoolProjects.PROJECT_ABC,
                        activity=activity, success=True)

        return result_info


class WordRefiner:
    """
    An improved word refinement system to tackle challenges with various open-source tools:
    Reference links:
    [1] Data Source Example
    [2] Model Implementation Details
    [3] Original Idea Source
    """

    def __init__(self, word_db_route=None):
        if word_db_route is None:
            directory_main = os.environ.get(ENV_WORD_DB)
            if directory_main is None:
                directory_main = DEFAULT_WORD_DB_PATH
            directory_main = Path(directory_main)
            if not directory_main.is_dir():
                raise FileNotFoundError(
                    f"Cannot locate directory: {directory_main}")

            word_stats_route = directory_main / "word-statistics" / "stats.json"
        else:
            word_stats_route = word_db_route

        assert word_stats_route.exists()

        with open(word_stats_route) as file_reader:
            word_statistics = json.load(file_reader)
        self.word_statistics = word_statistics

    @staticmethod
    def extract_words(input_text):
        return re.findall(r"\w+", input_text)

    def prob_metric(self, term):
        return -self.word_statistics.get(term, 0)

    def refine_word(self, term):
        return max(self.possible_terms(term), key=self.prob_metric)

    def possible_terms(self, term):
        return (
            self.recognized([term])
            or self.recognized(self.single_edit(term))
            or self.recognized(self.double_edit(term))
            or [term]
        )

    def recognized(self, terms):
        return set(t for t in terms if t in self.word_statistics)

    @staticmethod
    def single_edit(term):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        partitions = [(term[:i], term[i:]) for i in range(len(term) + 1)]
        removals = [L + R[1:] for L, R in partitions if R]
        swaps = [L + R[1] + R[0] + R[2:] for L, R in partitions if len(R) > 1]
        alterations = [L + char + R[1:]
                       for L, R in partitions if R for char in alphabet]
        additions = [L + char + R for L, R in partitions for char in alphabet]
        return set(removals + swaps + alterations + additions)

    def double_edit(self, term):
        return (secondary for primary in self.single_edit(term) for secondary in self.single_edit(primary))

    def __call__(self, input_text):
        extracted_words = self.extract_words(input_text)
        familiar_words = self.recognized(extracted_words)
        mapping = {word: self.refine_word(
            word) for word in extracted_words if word not in familiar_words}

        refined_text = input_text
        for original, refined in mapping.items():
            refined_text = re.sub(
                r"\b{}\b".format(re.escape(original)), refined, refined_text
            )
        return refined_text


class TextAnalyzer:
    """Generate features from given textual content."""

    def __init__(self):
        self.IGNORE_WORDS = set(stopwords.words("english"))
        self.spell_validator = SpellValidator()
        self.lang_model = spacy.load("en_core_web_lg")

    def overlap(self, data):
        def is_ignored(word):
            return word in self.IGNORE_WORDS

        source = data["source_tokens"]
        extract = data["extract_tokens"]
        if self.IGNORE_WORDS:
            source = list(filter(is_ignored, source))
            extract = list(filter(is_ignored, extract))
        return len(set(source) & set(extract))

    def seq_ngrams(self, sequence, n_val):
        grams = zip(*[sequence[idx:] for idx in range(n_val)])
        return [" ".join(gram) for gram in grams]

    def shared_ngrams(self, data, n_val):
        source_seq = data["source_tokens"]
        extract_seq = data["extract_tokens"]

        source_grams = set(self.seq_ngrams(source_seq, n_val))
        extract_grams = set(self.seq_ngrams(extract_seq, n_val))

        shared_grams = source_grams & extract_grams
        return len(shared_grams)

    def citation_count(self, data):
        extract = data["content"]
        source = data["source_content"]
        citations = re.findall(r'"([^"]*)"', extract)
        if citations:
            return [citation in source for citation in citations].count(True)
        return 0

    def typo_count(self, content):
        tokens = self.spell_validator.tokenize(content)
        recognized = self.spell_validator.known(tokens)
        unknown_tokens = [tok for tok in tokens if tok not in recognized]
        return len(unknown_tokens)

    def enrich_spellcheck(self, token_list):
        for token in token_list:
            if token not in self.spell_validator.token_probabilities:
                self.spell_validator.token_probabilities[token] = 0

    def extract_lemma(self, data, field, discard_stop=False, omit_punct=False):
        processed_data = data[field]
        out = []
        for entry in processed_data:
            if discard_stop and entry.is_stop:
                continue
            if omit_punct and entry.is_punct:
                continue
            out.append(entry.lemma_)
        return out

    def lemmatized_overlap(self, data):
        source_lemmas = self.extract_lemma(
            data, "source_content_processed", True, True)
        extract_lemmas = self.extract_lemma(
            data, "corrected_extract_processed", True, True)
        return len(set(source_lemmas) & set(extract_lemmas))

    def chunks_shared(self, data, discard_stop):
        source_chunks = self.extract_lemma(
            data, "source_content_processed", discard_stop, True)
        source_pairs = set(zip(source_chunks, source_chunks[1:]))

        extract_chunks = self.extract_lemma(
            data, "corrected_extract_processed", discard_stop, True)
        extract_pairs = set(zip(extract_chunks, extract_chunks[1:]))
        return len(source_pairs & extract_pairs)

    def named_entities(self, parsed):
        EXCLUDED_ENTITIES = {"ORDINAL", "CARDINAL", "PERCENT", "DATE", "TIME"}
        return {entity.text.strip() for entity in parsed.ents if entity.label_ not in EXCLUDED_ENTITIES}

    def entities_shared(self, data):
        source_entities = data["source_entities"]
        extract_entities = data["extract_entities"]
        return len(source_entities & extract_entities)

    def process(self, sources: pd.DataFrame, extracts: pd.DataFrame) -> pd.DataFrame:
        # Preprocessing for sources
        sources["source_tokens"] = sources["source_content"].apply(
            word_tokenize)
        sources["source_size"] = sources["source_tokens"].apply(len)
        sources["source_tokens"].apply(self.enrich_spellcheck)
        sources["source_content_processed"] = sources["source_content"].apply(
            self.lang_model)
        sources["source_entities"] = sources["source_content_processed"].apply(
            self.named_entities)

        # Preprocessing for extracts
        extracts["extract_tokens"] = extracts["content"].apply(word_tokenize)
        extracts["extract_size"] = extracts["extract_tokens"].apply(len)
        extracts["typos"] = extracts["content"].progress_apply(self.typo_count)
        extracts["corrected_extract"] = extracts["content"].progress_apply(
            self.spell_validator)
        extracts["corrected_extract_processed"] = extracts["corrected_extract"].progress_apply(
            self.lang_model)
        extracts["extract_entities"] = extracts["corrected_extract_processed"].apply(
            self.named_entities)

        # Merging sources and extracts
        combined_df = extracts.merge(sources, how="left", on="source_id")

        # Post-merge processing
        combined_df["size_ratio"] = combined_df["extract_size"] / \
            combined_df["source_size"]
        combined_df["overlap"] = combined_df.progress
