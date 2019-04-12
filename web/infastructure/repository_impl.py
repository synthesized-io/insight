import json
import uuid
from collections import namedtuple

from ..domain.repository import Repository, Directory


class InMemoryRepository(Repository):
    def __init__(self):
        self.mapping = {}

    def save(self, entity):
        if not entity.id:
            entity.id = str(uuid.uuid4())
        self.mapping[entity.id] = entity

    def get(self, entity_id):
        return self.mapping.get(entity_id, None)

    def find_by_props(self, prop_dict):
        result = []
        for _, entity in self.mapping.items():
            matches = True
            for prop, val in prop_dict.items():
                if getattr(entity, prop) != val:
                    matches = False
                    break
            if matches:
                result.append(entity)
        return result

    def delete(self, entity):
        self.mapping.pop(entity.id, None)


class SQLAlchemyRepository(Repository):
    def __init__(self, db, cls):
        self.db = db
        self.cls = cls

    def save(self, entity):
        self.db.session.add(entity)
        self.db.session.commit()

    def get(self, entity_id):
        return self.db.session.query(self.cls).get(entity_id)

    def find_by_props(self, prop_dict):
        q = self.db.session.query(self.cls)
        for attr, value in prop_dict.items():
            q = q.filter(getattr(self.cls, attr) == value)
        return q.all()

    def delete(self, entity):
        self.db.session.delete(entity)
        self.db.session.commit()


class JsonFileDirectory(Directory):
    def __init__(self, file_path, items_attribute):
        with open(file_path, 'r') as f:
            js = json.load(f)
            self.items = list(map(lambda d: namedtuple('X', d.keys())(*d.values()), js[items_attribute]))

    def list_items(self):
        return self.items
