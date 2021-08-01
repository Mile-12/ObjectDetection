from Main.api.image_detection_api import ImageApi


def initialize_routes(api):
    #POST: 
    api.add_resource(ImageApi,'/api/image')
