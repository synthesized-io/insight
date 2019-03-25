from ..app import db
from datetime import datetime
from enum import Enum


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
    meta = db.Column(db.LargeBinary, nullable=False)
    syntheses = db.relationship("Synthesis", cascade="all, delete-orphan", lazy='select')
    reports = db.relationship("Report", cascade="all, delete-orphan", lazy='select')

    def __str__(self):
        return '<Dataset {}>'.format(self.id)


class Synthesis(db.Model, AuditMixin):
    id = db.Column(db.Integer, primary_key=True)
    blob = db.Column(db.LargeBinary, nullable=False)
    meta = db.Column(db.LargeBinary, nullable=False)
    size = db.Column(db.Integer, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey(Dataset.id), nullable=False)

    def __str__(self):
        return '<Synthesis {}>'.format(self.id)


class Report(db.Model, AuditMixin):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey(Dataset.id), nullable=False)
    items = db.relationship("ReportItem", cascade="all, delete-orphan", lazy='select')

    def __str__(self):
        return '<Report {}>'.format(self.id)


class ReportItemType(Enum):
    CORRELATION = 1
    MODELLING = 2


class ReportItem(db.Model, AuditMixin):
    id = db.Column(db.Integer, primary_key=True)
    ord = db.Column(db.Integer, nullable=False)
    report_id = db.Column(db.Integer, db.ForeignKey(Report.id), nullable=False)
    item_type = db.Column(db.Enum(ReportItemType), nullable=False)
    settings = db.Column(db.LargeBinary)
    results = db.Column(db.LargeBinary)

    def __str__(self):
        return '<ReportItem {}>'.format(self.id)
