import os
from io import StringIO
from typing import List

import pandas as pd
from flask import current_app

from ..domain.dataset_meta import compute_dataset_meta
from ..domain.model import Dataset
from ..domain.model import ProjectTemplate
from ..domain.repository import Directory, Repository


class ProjectTemplates:
    def __init__(self, template_dir: Directory, dataset_repo: Repository):
        self.template_dir = template_dir
        self.dataset_repo = dataset_repo

    def create_dataset_from_template(self, user_id: int, template_id: int) -> Dataset:
        templates: List[ProjectTemplate] = self.template_dir.list_items()
        templates = [t for t in templates if t.id == template_id]
        if len(templates) == 0:
            raise Exception('Template not found: ' + str(template_id))

        template = templates[0]

        data = pd.read_csv(os.path.join(current_app.root_path, 'project_templates', template.file))
        raw_data = StringIO()
        data.to_csv(raw_data, index=False, encoding='utf-8')
        meta = compute_dataset_meta(data)
        blob = raw_data.getvalue().encode('utf-8')

        dataset = Dataset(user_id=user_id, title=template.title, description=template.description, blob=blob)
        dataset.set_meta_from_object(meta)

        self.dataset_repo.save(dataset)

        return dataset
