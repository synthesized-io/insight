import gc
import logging
from collections import OrderedDict
from enum import Enum
from io import BytesIO
from queue import Queue
from threading import Lock, Thread
from typing import Optional, Union, Tuple

import pandas as pd

from synthesized.core import BasicSynthesizer, Synthesizer
from ..domain.dataset_meta import recompute_dataset_meta, DatasetMeta
from ..domain.model import Dataset
from ..domain.repository import Repository

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    NO_MODEL = 1
    TRAINING = 2
    FAILED = 3
    READY = 4


class Preview:
    def __init__(self, meta: DatasetMeta, iteration: int, total_iterations: int):
        self.meta = meta
        self.iteration = iteration
        self.total_iterations = total_iterations


class SynthesizerManager:
    def __init__(self, dataset_repo: Repository, max_models: int, preview_size=1000, yield_every=250,
                 num_iterations=2500, max_sample_size=100000):
        self.dataset_repo = dataset_repo
        self.max_models = max_models
        self.preview_size = preview_size
        self.yield_every = yield_every
        self.num_iterations = num_iterations
        self.max_sample_size = max_sample_size
        self.cache = OrderedDict()
        self.cache_lock = Lock()
        self.preview_cache = {}
        self.preview_cache_lock = Lock()
        self.requests = set()
        self.requests_lock = Lock()
        self.request_queue = Queue()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            dataset_id = self.request_queue.get()
            synthesizer_or_error = None
            for i, train_result in self._train_model(dataset_id):
                if isinstance(train_result, Synthesizer):
                    try:
                        dataset: Dataset = self.dataset_repo.get(dataset_id)
                        if not dataset:
                            break
                        old_meta = dataset.get_meta_as_object()
                        data = train_result.synthesize(self.preview_size)
                        meta = recompute_dataset_meta(data, old_meta)
                        preview = Preview(meta, i, self.num_iterations)
                        with self.preview_cache_lock:
                            self.preview_cache[dataset_id] = preview
                    except Exception as e:
                        logger.error(e)
                synthesizer_or_error = train_result
            with self.cache_lock:
                if len(self.cache) == self.max_models:
                    logger.info("popping first item from cache")
                    old_model = self.cache.popitem(last=False)
                    if isinstance(old_model, Synthesizer):
                        try:
                            old_model.__exit__(None, None, None)
                        except Exception as e:
                            logger.error(e)
                    del old_model
                    gc.collect()
                self.cache[dataset_id] = synthesizer_or_error
            with self.requests_lock:
                self.requests.remove(dataset_id)
            with self.preview_cache_lock:
                if dataset_id in self.preview_cache:
                    del self.preview_cache[dataset_id]

    def _train_model(self, dataset_id) -> Tuple[int, Union[Synthesizer, Exception]]:
        dataset: Dataset = self.dataset_repo.get(dataset_id)
        if not dataset:
            yield 0, Exception('Could not find dataset by id=' + str(dataset_id))

        disabled_columns = dataset.get_settings_as_dict().get('disabled_columns', [])

        data = pd.read_csv(BytesIO(dataset.blob), encoding='utf-8')
        data = data.drop(disabled_columns, axis=1)
        sample_size = min(len(data), self.max_sample_size)
        data = data.sample(sample_size)
        data = data.dropna()

        logger.info('start model training')
        try:
            synthesizer = BasicSynthesizer(data=data)
            synthesizer.__enter__()
            for iter in synthesizer.learn_async(num_iterations=self.num_iterations, data=data, yield_every=self.yield_every):
                yield (iter, synthesizer)
            logger.info('model has been trained')
        except Exception as e:
            logger.exception(e)
            yield 0, e

    def train_async(self, dataset_id) -> None:
        with self.requests_lock:
            if dataset_id not in self.requests:
                self.requests.add(dataset_id)
                self.request_queue.put(dataset_id)

    def get_status(self, dataset_id) -> ModelStatus:
        with self.cache_lock:
            if dataset_id in self.cache:
                if isinstance(self.cache[dataset_id], Exception):
                    return ModelStatus.FAILED
                else:
                    return ModelStatus.READY
        with self.requests_lock:
            if dataset_id in self.requests:
                return ModelStatus.TRAINING
        return ModelStatus.NO_MODEL

    def get_preview(self, dataset_id) -> Preview:
        with self.preview_cache_lock:
            return self.preview_cache.get(dataset_id, None)

    def get_model(self, dataset_id) -> Optional[Synthesizer]:
        with self.cache_lock:
            synthesizer_or_error = self.cache.get(dataset_id, None)
            if isinstance(synthesizer_or_error, Exception):
                return None
            else:
                return synthesizer_or_error
