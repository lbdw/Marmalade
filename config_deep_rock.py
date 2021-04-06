import tensorflow as tf

def config_device():
    # Print available devices
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)

    cpus = tf.config.experimental.list_physical_devices('CPU')
    print(cpus)

    # Configure logical device
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.set_visible_devices(gpus[1], 'GPU')

            # Restrict TensorFlow to use the CPU
            # tf.config.set_visible_devices([], 'GPU')


            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                assert tf.config.experimental.get_memory_growth(gpu)

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)