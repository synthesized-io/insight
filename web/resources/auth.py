from flask import current_app
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_refresh_token_required, get_jwt_identity
from flask_restful import Resource, reqparse, abort

from ..model import User


class LoginResource(Resource):
    def __init__(self, **kwargs):
        self.authenticator = kwargs['authenticator']

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('username', type=str, required=True)
        parser.add_argument('password', type=str, required=True)
        args = parser.parse_args()

        username = args['username']
        password = args['password']

        user = self.authenticator.authenticate(username, password)
        if not user:
            return {"message": "Bad username or password"}, 401

        ret = {
            'access_token': create_access_token(identity=user.id),
            'refresh_token': create_refresh_token(identity=user.id)
        }
        return ret, 200


class RefreshResource(Resource):
    decorators = [jwt_refresh_token_required]

    def post(self):
        current_user = get_jwt_identity()
        ret = {
            'access_token': create_access_token(identity=current_user)
        }
        return ret, 200


class UsersResource(Resource):
    def __init__(self, **kwargs):
        self.user_repo = kwargs['user_repo']
        self.bcrypt = kwargs['bcrypt']

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('username', type=str, required=True)
        parser.add_argument('password', type=str, required=True)
        args = parser.parse_args()

        username = args['username']
        password = args['password']

        current_app.logger.info('registering user {}'.format(username))

        users = self.user_repo.find_by_props({'username': username})
        if len(users) > 0:
            current_app.logger.info('found existing user {}'.format(users[0]))
            abort(409, message='User with username={} already exists'.format(username))

        user = User(username=username, password=self.bcrypt.generate_password_hash(password).hex())
        self.user_repo.save(user)
        current_app.logger.info('created a user {}'.format(user))

        ret = {
            'access_token': create_access_token(identity=user.id),
            'refresh_token': create_refresh_token(identity=user.id)
        }
        return ret, 200
