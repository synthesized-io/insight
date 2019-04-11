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
        parser.add_argument('email', type=str, required=True)
        parser.add_argument('password', type=str, required=True)
        args = parser.parse_args()

        email = args['email']
        password = args['password']

        user = self.authenticator.authenticate(email, password)
        if not user:
            return {"message": "Bad email or password"}, 401

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
        parser.add_argument('email', type=str, required=True)
        parser.add_argument('password', type=str, required=True)

        parser.add_argument('first_name', type=str)
        parser.add_argument('last_name', type=str)
        parser.add_argument('phone_number', type=str)
        parser.add_argument('job_title', type=str)
        parser.add_argument('company', type=str)

        args = parser.parse_args()

        invite_code = args['invite_code']
        email = args['email']
        password = args['password']

        first_name = args['first_name']
        last_name = args['last_name']
        phone_number = args['phone_number']
        job_title = args['job_title']
        company = args['company']

        current_app.logger.info('registering user {}'.format(email))

        if not check_invite_code(invite_code, email, current_app.config['INVITE_KEY']):
            abort(400, message='Invalid invite code')

        used_invite = self.used_invite_repo.get(invite_code)
        if used_invite:
            abort(409, message='Invite code {} has already been used'.format(invite_code))

        users = self.user_repo.find_by_props({'email': email})
        if len(users) > 0:
            current_app.logger.info('found existing user {}'.format(users[0]))
            abort(409, message='User with email={} already exists'.format(email))

        user = User(email=email,
                    password=self.bcrypt.generate_password_hash(password).hex(),
                    first_name=first_name,
                    last_name=last_name,
                    phone_number=phone_number,
                    job_title=job_title,
                    company=company)
        self.user_repo.save(user)
        current_app.logger.info('created a user {}'.format(user))

        used_invite = UsedInvite(code=invite_code, user_id=user.id)
        self.used_invite_repo.save(used_invite)

        ret = {
            'access_token': create_access_token(identity=user.id),
            'refresh_token': create_refresh_token(identity=user.id)
        }
        return ret, 200