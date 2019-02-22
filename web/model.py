from .app import db


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    blob = db.Column(db.Text, nullable=False)
    meta = db.Column(db.Text, nullable=False)
    syntheses = db.relationship("Synthesis", cascade="all, delete-orphan", lazy='select')

    def __repr__(self):
        return '<Dataset {}>'.format(self.id)


class Synthesis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    blob = db.Column(db.Text, nullable=False)
    size = db.Column(db.Integer, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey(Dataset.id), nullable=False)

    def __repr__(self):
        return '<Synthesis {}>'.format(self.id)
