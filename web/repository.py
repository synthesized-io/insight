import uuid


class Repository:
    def save(self, entity):
        pass

    def find(self, entity_id):
        pass

    def delete(self, entity):
        pass


class InMemoryRepository(Repository):
    def __init__(self):
        self.mapping = {}

    def save(self, entity):
        if not entity.id:
            entity.id = str(uuid.uuid4())
        self.mapping[entity.id] = entity

    def find(self, entity_id):
        return self.mapping.get(entity_id, None)

    def delete(self, entity):
        self.mapping.pop(entity.id, None)


class SQLAlchemyRepository(Repository):
    def __init__(self, db, cls):
        self.db = db
        self.cls = cls

    def save(self, entity):
        self.db.session.add(entity)
        self.db.session.commit()

    def find(self, entity_id):
        return self.db.session.query(self.cls).get(entity_id)

    def delete(self, entity):
        self.db.session.delete(entity)
        self.db.session.commit()
