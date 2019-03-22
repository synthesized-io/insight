from flask_restful import Resource


class StatusResource(Resource):
    def get(self):
        return {'success': True}
