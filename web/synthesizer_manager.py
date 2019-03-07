import logging
from enum import Enum
from io import BytesIO
from queue import Queue
from threading import Lock, Thread

import pandas as pd

from synthesized.core import BasicSynthesizer
from .repository import Repository

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    NO_MODEL = 1
    TRAINING = 2
    READY = 3


class SynthesizerManager:
    def __init__(self, dataset_repo: Repository):
        self.cache = {}
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
            synthesizer = self._train_model(dataset_id)
            if not synthesizer:
                continue
            with self.cache_lock:
                self.cache[dataset_id] = synthesizer
            with self.requests_lock:
                self.requests.remove(dataset_id)

    def _train_model(self, dataset_id):
        dataset = self.dataset_repo.get(dataset_id)
        if not dataset:
            logger.info('could not find dataset by id=' + dataset_id)
            return

        data = pd.read_csv(BytesIO(dataset.blob), encoding='utf-8')
        data = data.dropna()

        logger.info('starting model training')
        try:
            synthesizer = BasicSynthesizer(data=data)
            synthesizer.__enter__()
            synthesizer.learn(data=data)
            logger.info('model has been trained')
            return synthesizer
        except Exception as e:
            logger.exception(e)

    def train_async(self, dataset_id):
        with self.requests_lock:
            if dataset_id not in self.requests:
                self.requests.add(dataset_id)
                self.request_queue.put(dataset_id)

    def get_status(self, dataset_id):
        with self.cache_lock:
            if dataset_id in self.cache:
                return ModelStatus.READY
        with self.requests_lock:
            if dataset_id in self.requests:
                return ModelStatus.TRAINING
        return ModelStatus.NO_MODEL

    def get_model(self, dataset_id):
        with self.cache_lock:
            return self.cache.get(dataset_id, None)
