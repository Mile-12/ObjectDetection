"""This is init module."""
from flask_restful import Api
from flask import Flask, Response, request, jsonify
from flask_bcrypt import Bcrypt
import os
from Main.Routes.routes import initialize_routes
from flask_cors import CORS
# Place where app is defined
app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = 'Main/api/uploads'
api = Api(app)
#CORS(app, resources={r'/api/*':{"origins": "*"}},allow_headers=[
 #   "Content-Type", "Authorization", "Access-Control-Allow-Methods"])
#CORS(app)

initialize_routes(api)
bcrypt = Bcrypt(app)
