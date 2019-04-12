class Authenticator:
    def __init__(self, user_repo, bcrypt):
        self.user_repo = user_repo
        self.bcrypt = bcrypt

    def authenticate(self, email, password):
        users = self.user_repo.find_by_props({'email': email})
        if len(users) > 0:
            password_hash = bytes.fromhex(users[0].password)
            if self.bcrypt.check_password_hash(password_hash, password):
                return users[0]
