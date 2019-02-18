class Entity:
    def __init__(self, entity_id):
        self.entity_id = entity_id


class Dataset(Entity):
    def __init__(self, dataset_id, blob, meta):
        super().__init__(dataset_id)
        self.blob = blob
        self.meta = meta


class Synthesis(Entity):
    def __init__(self, synthesis_id, dataset_id, blob, size):
        super().__init__(synthesis_id)
        self.dataset_id = dataset_id
        self.blob = blob
        self.size = size
