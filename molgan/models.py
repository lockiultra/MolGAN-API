# from pony import *

# db = Database()

# class User(db.Entity):
#     id = PrimaryKey(int, auto=True)
#     username = Required(str)
#     password = Required(str)
#     email = Required(str)

# class Molecule(db.Entity):
#     id = PrimaryKey(int, auto=True)
#     name = Required(str)
#     smiles = Required(str)
#     disease_probs = Required(str)

# class Session(db.Entity):
#     id = PrimaryKey(int, auto=True)
#     token = Required(str)

# class Token(db.Entity):
#     id = PrimaryKey(int, auto=True)
#     token = Required(str)

# db.bind(provider='postgresql', filename='users.db', create_db=True)