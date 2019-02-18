import uuid


class Repository:
    def save(self, entity):
        pass

    def find(self, entity_id):
        pass

    def delete(self, entity_id):
        pass


class InMemoryRepository(Repository):
    def __init__(self):
        self.mapping = {}

    def save(self, entity):
        if not entity.entity_id:
            entity.entity_id = str(uuid.uuid4())
        self.mapping[entity.entity_id] = entity

    def find(self, entity_id):
        return self.mapping.get(entity_id, None)

    def delete(self, entity_id):
        self.mapping.pop(entity_id, None)
