import logging
from collections import OrderedDict
from enum import Enum
from io import BytesIO
from queue import Queue
from threading import Lock, Thread
import gc

import pandas as pd

from synthesized.core import BasicSynthesizer, Synthesizer
from .repository import Repository

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    NO_MODEL = 1
    TRAINING = 2
    FAILED = 3
    READY = 4


class SynthesizerManager:
    def __init__(self, dataset_repo: Repository, max_models):
        self.cache = OrderedDict()
        self.max_models = max_models
        self.requests = set()
        self.cache_lock = Lock()
        self.requests_lock = Lock()
        self.dataset_repo = dataset_repo
        self.request_queue = Queue()
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            dataset_id = self.request_queue.get()
            synthesizer_or_error = self._train_model(dataset_id)
            if not synthesizer_or_error:
                continue
            with self.cache_lock:
                if len(self.cache) == self.max_models:
                    logger.info("popping first item from cache")
                    old_model = self.cache.popitem(last=False)
                    if isinstance(old_model, Synthesizer):
                        try:
                            old_model.__exit__(None, None, None)
                        except Exception as e:
                            logger.error(e)
                    old_model = None  # loose ref to the object
                    gc.collect()
                self.cache[dataset_id] = synthesizer_or_error
            with self.requests_lock:
                self.requests.remove(dataset_id)

    def _train_model(self, dataset_id):
        dataset = self.dataset_repo.get(dataset_id)
        if not dataset:
            logger.info('could not find dataset by id=' + dataset_id)
            return

        data = pd.read_csv(BytesIO(dataset.blob), encoding='utf-8')
        data = data.dropna()

        logger.info('start model training')
        try:
            synthesizer = BasicSynthesizer(data=data)
            synthesizer.__enter__()
            synthesizer.learn(data=data)
            logger.info('model has been trained')
            return synthesizer
        except Exception as e:
            logger.exception(e)
            return e

    def train_async(self, dataset_id):
        with self.requests_lock:
            if dataset_id not in self.requests:
                self.requests.add(dataset_id)
                self.request_queue.put(dataset_id)

    def get_status(self, dataset_id):
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

    def get_model(self, dataset_id):
        with self.cache_lock:
            synthesizer_or_error = self.cache.get(dataset_id, None)
            if isinstance(str, Exception):
                return None
            else:
                return synthesizer_or_error
