import runway
from runway.data_types import number, text, image, array, image_bounding_box
from example_model import FaceTracker

setup_options = {
    # 'truncation': number(min=1, max=10, step=1, default=5, description='Example input.'),
    # 'seed': number(min=0, max=1000000, description='A seed used to initialize the model.')
}
@runway.setup(options=setup_options)
def setup(opts):
    # msg = '[SETUP] Ran with options: seed = {}, truncation = {}'
    # print(msg.format(opts['seed'], opts['truncation']))
    model = FaceTracker(opts)
    return model

@runway.command(name='find_faces',
                inputs={ 'input': image(description="The input image to analyze") },
                outputs={ 'ids': array(number, description="ID's of found faces"),'boxes': array(image_bounding_box, description="bounding boxes of found faces") },
                description='Look for faces in the image')
def find_faces(model, args):
    # print('[GENERATE] Ran with caption value "{}"'.format(args['caption'])) 
    # soutput_image = model.run_on_input(args['caption'])
    
    output = model.process(args['input'])

    return {
        'ids': [o["index"] for o in output], 
        'boxes': [o["box"] for o in output]
    }

if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000, debug=True)


 