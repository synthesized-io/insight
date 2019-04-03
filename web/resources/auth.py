from flask import current_app
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_refresh_token_required, get_jwt_identity
from flask_restful import Resource, reqparse, abort

from ..application.authenticator import Authenticator
from ..application.invites import check_invite_code
from ..domain.model import User, UsedInvite
from ..domain.repository import Repository


class LoginResource(Resource):
    def __init__(self, **kwargs):
        self.authenticator: Authenticator = kwargs['authenticator']

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
        self.user_repo: Repository = kwargs['user_repo']
        self.used_invite_repo: Repository = kwargs['used_invite_repo']
        self.bcrypt: Bcrypt = kwargs['bcrypt']

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('invite_code', type=str, required=True)
        parser.add_argument('username', type=str, required=True)
        parser.add_argument('password', type=str, required=True)
        args = parser.parse_args()

        invite_code = args['invite_code']
        username = args['username']
        password = args['password']

        current_app.logger.info('registering user {}'.format(username))

        used_invite = self.used_invite_repo.get(invite_code)
        if used_invite:
            abort(409, message='Invite code {} has already been used'.format(invite_code))

        if not check_invite_code(invite_code, current_app.config['INVITE_KEY']):
            abort(400, message='Invite code is invalid')

        users = self.user_repo.find_by_props({'username': username})
        if len(users) > 0:
            current_app.logger.info('found existing user {}'.format(users[0]))
            abort(409, message='User with username={} already exists'.format(username))

        user = User(username=username, password=self.bcrypt.generate_password_hash(password).hex())
        self.user_repo.save(user)
        current_app.logger.info('created a user {}'.format(user))

        used_invite = UsedInvite(code=invite_code, user_id=user.id)
        self.used_invite_repo.save(used_invite)

        ret = {
            'access_token': create_access_token(identity=user.id),
            'refresh_token': create_refresh_token(identity=user.id)
        }
        return ret, 200
