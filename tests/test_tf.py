import tensorflow as tf

print('--- TF INFO ---')
print(f'TF Version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPU Available: {len(gpus) > 0}')
print(f'Number of GPUs: {len(gpus)}')
print('---------------')
