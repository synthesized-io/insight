from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_restful import Resource

from ..application.project_templates import ProjectTemplates
from ..domain.repository import Directory


class ProjectTemplatesResource(Resource):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.template_directory: Directory = kwargs['template_directory']

    def get(self):
        items = self.template_directory.list_items()
        return [{
            'id': item.id,
            'title': item.title,
            'description': item.description
        } for item in items]


class DatasetFromTemplateResource(Resource):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.project_templates: ProjectTemplates = kwargs['project_templates']

    def post(self, template_id):
        template_id = int(template_id)
        dataset = self.project_templates.create_dataset_from_template(user_id=get_jwt_identity(), template_id=template_id)
        return {'dataset_id': dataset.id}, 201, {'Location': '/datasets/{}'.format(dataset.id)}
