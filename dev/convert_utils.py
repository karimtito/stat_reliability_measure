import tf2onnx
from onnx2torch import convert
import tensorflow as tf

def tf2torch(tf_model, model_name,device='cpu',save=True):
    """converts tf model to torch model via ONNX """
    onnx_path = model_name+'.onnx'
    onnx_tuple = tf2onnx.convert.from_keras(tf_model,
                        output_path=onnx_path)
    torch_model = convert(onnx_path).to(device)
    return torch_model

def load_torch_from_keras(h5_path, logit_model=True, device='cpu'):
    """loads torch model from keras h5 file"""
    tf_model = tf.keras.models.load_model(h5_path,compile=False)
    # if the last layer is softmax, remove it to get logits
    if logit_model and 'activation' in tf_model.layers[-1].get_config().keys():
        if tf_model.layers[-1].get_config()['activation']=='softmax':
            tf_model = tf.keras.models.Model(inputs=tf_model.input, outputs=tf_model.layers[-2].output)
    model_name = h5_path.split('/')[-1].split('.')[0]
    torch_model = tf2torch(tf_model, model_name, device=device)
    return torch_model 