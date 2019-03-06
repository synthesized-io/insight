from .app import db
from datetime import datetime


class AuditMixin(object):
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class User(db.Model, AuditMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.Text, nullable=False, unique=True)
    password = db.Column(db.Text, nullable=False)

    def __str__(self):
        return "<User {}>".format(self.id)


class Dataset(db.Model, AuditMixin):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey(User.id), nullable=False)
    title = db.Column(db.Text)
    description = db.Column(db.Text)
    blob = db.Column(db.LargeBinary, nullable=False)
    meta = db.Column(db.Text, nullable=False)
    syntheses = db.relationship("Synthesis", cascade="all, delete-orphan", lazy='select')

    def __str__(self):
        return '<Dataset {}>'.format(self.id)


class Synthesis(db.Model, AuditMixin):
    id = db.Column(db.Integer, primary_key=True)
    blob = db.Column(db.LargeBinary, nullable=False)
    size = db.Column(db.Integer, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey(Dataset.id), nullable=False)

    def __str__(self):
        return '<Synthesis {}>'.format(self.id)
