from flask import Flask, request
from config import Config



def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    with app.app_context():
        from api.app import api
        app.register_blueprint(api, url_prefix="/api", template_folder='templates', static_folder='static')
        
        @app.after_request
        def after_request(response):
            request.get_data()
            return response
    
    return app

app = create_app()
