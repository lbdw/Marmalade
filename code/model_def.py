import tensorflow as tf
import numpy as np


def build_model():
    inputs = tf.keras.Input(shape=(2,), name="input_vector")
    outputs = tf.keras.layers.Dense(2, activation='linear', name="predictions")(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def build_linear_model(input_len, output_len):
    inputs = tf.keras.Input(shape=(input_len,), name="input_vector")
    outputs = tf.keras.layers.Dense(output_len, activation='linear', name="predictions")(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


# select_input_features = ['SiO2','TiO2','Al2O3','FeO','MgO','CaO','Na2O','K2O', 'P(GPa)', 'T(C)', 'log fO2', 'H2O', 'Original indices']
def build_x_model_SiTiAlFeMgCaNaKH_reg(input_len, output_len):

    inputs = tf.keras.Input(shape=(input_len,), name="input_vector")

    ##  Si  #######################################################
    reg_L = [0.01, 0.01, 0.1]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[0]))(inputs)
    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[1]))(inputs)

    Si = tf.keras.layers.Dense(1, activation='relu', name="Si",
                               kernel_regularizer=tf.keras.regularizers.l2(reg_L[2])
                               )(x)

    ##  Al  ####################################################### ok
    reg_L = [0.01, 0.01, 0.1]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[0]))(
        inputs)
    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[1]))(
        inputs)

    Al = tf.keras.layers.Dense(1, activation='relu', name="Al",
                               kernel_regularizer=tf.keras.regularizers.l2(reg_L[2])
                               )(x)

    ##  Fe  ####################################################### nice
    reg_L = [0.01, 0.01, 0.1]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[0]))(
        inputs)
    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[1]))(
        inputs)

    Fe = tf.keras.layers.Dense(1, activation='relu', name="Fe",
                               kernel_regularizer=tf.keras.regularizers.l2(reg_L[2])
                               )(x)

    ##  Mg  ####################################################### nice
    reg_L = [0.01, 0.01, 0.1]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[0]))(
        inputs)
    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[1]))(
        inputs)

    Mg = tf.keras.layers.Dense(1, activation='relu', name="Mg",
                               kernel_regularizer=tf.keras.regularizers.l2(reg_L[2])
                               )(x)

    ##  Ca  ####################################################### nice
    reg_L = [0.01, 0.01, 0.1]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[0]))(
        inputs)
    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[1]))(
        inputs)

    Ca = tf.keras.layers.Dense(1, activation='relu', name="Ca",
                               kernel_regularizer=tf.keras.regularizers.l2(reg_L[2])
                               )(x)

    ##  Ti  ####################################################### ok
    reg_L = [0.01, 0.01, 0.1]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[0]))(
        inputs)
    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[1]))(
        inputs)

    Ti = tf.keras.layers.Dense(1, activation='relu', name="Ti",
                               kernel_regularizer=tf.keras.regularizers.l2(reg_L[2])
                               )(x)

    ##  Na  ####################################################### ok
    reg_L = [0.01, 0.01, 0.1]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[0]))(
        inputs)
    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[1]))(
        inputs)

    Na = tf.keras.layers.Dense(1, activation='relu', name="Na",
                               kernel_regularizer=tf.keras.regularizers.l2(reg_L[2])
                               )(x)

    ##  K  ####################################################### nice
    reg_L = [0.01, 0.01, 0.1]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[0]))(
        inputs)
    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[1]))(
        inputs)

    K = tf.keras.layers.Dense(1, activation='relu', name="K",
                               kernel_regularizer=tf.keras.regularizers.l2(reg_L[2])
                               )(x)

    ##  H2O  ####################################################### nice
    reg_L = [0.01, 0.01, 0.1]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[0]))(
        inputs)
    x = tf.keras.layers.Dense(layer_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_L[1]))(
        inputs)

    H2O = tf.keras.layers.Dense(1, activation='relu', name="H2O",
                               kernel_regularizer=tf.keras.regularizers.l2(reg_L[2])
                               )(x)


    #########################################################

    outputs = tf.keras.layers.Concatenate(name="predictions")([Si, Al, Fe, Mg, Ca, Ti, Na, K, H2O])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def build_x_model_SiTiAlFeMgCaNaKH(input_len, output_len):

    inputs = tf.keras.Input(shape=(input_len,), name="input_vector")

    ##  Si  #######################################################
    drop_rate = [0, 0.2, 0.4]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(drop_rate[0])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[1])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[2])(x)

    Si = tf.keras.layers.Dense(1, activation='relu', name="Si")(x)

    ##  Ti  ####################################################### ok
    drop_rate = [0, 0.2, 0.4]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(drop_rate[0])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[1])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[2])(x)

    Ti = tf.keras.layers.Dense(1, activation='relu', name="Ti")(x)

    ##  Al  ####################################################### ok
    drop_rate = [0.0, 0.2, 0.4]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(drop_rate[0])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[1])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[2])(x)

    Al = tf.keras.layers.Dense(1, activation='relu', name="Al")(x)

    ##  Fe  ####################################################### nice
    drop_rate = [0, 0.2, 0.4]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(drop_rate[0])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[1])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[2])(x)

    Fe = tf.keras.layers.Dense(1, activation='relu', name="Fe")(x)

    ##  Mg  ####################################################### nice
    drop_rate = [0, 0.2, 0.4]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(drop_rate[0])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[1])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[2])(x)

    Mg= tf.keras.layers.Dense(1, activation='relu', name="Mg")(x)

    ##  Ca  ####################################################### nice
    drop_rate = [0, 0.2, 0.4]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(drop_rate[0])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[1])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[2])(x)

    Ca = tf.keras.layers.Dense(1, activation='relu', name="Ca")(x)

    ##  Na  ####################################################### ok
    drop_rate = [0, 0.2, 0.2]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(drop_rate[0])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[1])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[2])(x)

    Na = tf.keras.layers.Dense(1, activation='relu', name="Na")(x)

    ##  K  ####################################################### nice
    drop_rate = [0, 0.2, 0.4]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(drop_rate[0])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[1])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[2])(x)

    K = tf.keras.layers.Dense(1, activation='relu', name="K")(x)

    ##  H2O  ####################################################### nice
    drop_rate = [0, 0.2, 0.4]
    layer_units = 32

    x = tf.keras.layers.Dense(layer_units, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(drop_rate[0])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[1])(x)

    x = tf.keras.layers.Dense(layer_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop_rate[2])(x)

    H2O = tf.keras.layers.Dense(1, activation='relu', name="H2O")(x)


    #########################################################

    outputs = tf.keras.layers.Concatenate(name="predictions")([Si, Ti, Al, Fe, Mg, Ca, Na, K, H2O])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model



def build_xann_model(input_len):
    def custom_layer(tensor):
        exp_t = tf.exp(tensor)
        return exp_t / tf.reduce_sum(exp_t, axis=1, keepdims=True) * 100

    inputs = tf.keras.Input(shape=(input_len,), name="input_vector")

    x_units = 64
    y_units = 16
    z_units = 64

    L_x = np.array([0.01, 0.01, 0.01])
    L_y = np.array([0.01])
    L_z = np.array([0.01, 0.01, 0.01, 0.01])

    # the branch for anhydrous
    x1 = tf.keras.layers.Dense(x_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_x[0]))(inputs)
    x2 = tf.keras.layers.Dense(x_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_x[1]))(x1)
    x = tf.keras.layers.Dense(x_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_x[2]))(x2)

    # y = tf.keras.layers.Dense(y_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_y[0]))(x)

    Si = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.))(x) #Si
    Ti = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.))(x) # Ti
    Al = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.))(x)
    Fe = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.))(x)
    Mg = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.))(x)
    Ca = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.))(x)
    Na = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0))(x)
    K = tf.keras.layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0))(x)

    anhydrous_comp = tf.keras.layers.Concatenate(name="anhydrous_comp")([Si, Ti, Al, Fe, Mg, Ca, Na, K])

    anhydrous_comp_100 = tf.keras.layers.Lambda(custom_layer, name="anhydrous_comp_100")(anhydrous_comp)

    # the branch for H20
    z = tf.keras.layers.Dense(z_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_z[0]))(inputs)
    z = tf.keras.layers.Dense(z_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_z[1]))(z)
    z = tf.keras.layers.Dense(z_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_z[2]))(z)

    water = tf.keras.layers.Dense(1, activation='relu', name="water",
                                  kernel_regularizer=tf.keras.regularizers.l2(L_z[3]))(z)

    # outputs & model
    outputs = tf.keras.layers.Concatenate(name="predictions")([anhydrous_comp_100, water])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def build_xann_model_fine(input_len, output_len):

    def custom_layer(tensor):
        exp_t = tf.exp(tensor)
        return exp_t/tf.reduce_sum(exp_t, axis=1, keepdims=True)*100

    inputs = tf.keras.Input(shape=(input_len,), name="input_vector")

    x_units = 64
    y_units = 64
    z_units = 64

    L_x = np.array([0.001, 0.001, 0.001, 0.1])
    L_y = np.array([0.001, 0.001, 0.001, 0.1])
    L_z = np.array([0.001, 0.001, 0.001, 0.1])


    # the branch for SiAlFeMgCa
    x = tf.keras.layers.Dense(x_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_x[0]))(inputs)
    x = tf.keras.layers.Dense(x_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_x[1]))(x)
    x = tf.keras.layers.Dense(x_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_x[2]))(x)
    x = tf.keras.layers.Dense(output_len - 1 - 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_x[3]))(x)

    # the branch for TiNaK H
    y = tf.keras.layers.Dense(y_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_y[0]))(inputs)
    y = tf.keras.layers.Dense(y_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_y[1]))(y)
    y = tf.keras.layers.Dense(y_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_y[2]))(y)
    y = tf.keras.layers.Dense(3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_y[3]))(y)

    # merge SiAlFeMgCa and TiNaK
    anhydrous_comp_pred = tf.keras.layers.Concatenate(name="x_y")([x, y])
    anhydrous_comp_pred100 = tf.keras.layers.Lambda(custom_layer, name="anhydrous_comp_pred100")(anhydrous_comp_pred)

    # the branch for H20
    z = tf.keras.layers.Dense(z_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_z[0]))(inputs)
    z = tf.keras.layers.Dense(z_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_z[1]))(z)
    z = tf.keras.layers.Dense(z_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L_z[2]))(z)

    water = tf.keras.layers.Dense(1, activation='relu', name="water", kernel_regularizer=tf.keras.regularizers.l2(L_z[3]))(z)

    # outputs & model
    outputs = tf.keras.layers.Concatenate(name="predictions")([anhydrous_comp_pred100, water])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def build_ann_model(input_len, output_len):
    inputs = tf.keras.Input(shape=(input_len,), name="input_vector")

    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    # x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)

    # x = tf.keras.layers.Dense(64, activation='relu')(x)

    # x = tf.keras.layers.Dense(128, activation='relu')(x)
    # x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(output_len, activation='sigmoid', name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def build_ann_model_simple(input_len, output_len):
    inputs = tf.keras.Input(shape=(input_len,), name="input_vector")

    x = tf.keras.layers.Dense(32, activation='relu')(inputs)
    # x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(32, activation='relu')(x)

    # x = tf.keras.layers.Dense(64, activation='relu')(x)

    # x = tf.keras.layers.Dense(128, activation='relu')(x)
    # x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(output_len, activation='sigmoid', name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def norm_loss(y_true,y_pred):

    index = ~tf.math.is_nan(y_true)
    y_true = tf.boolean_mask(y_true, index)
    y_pred = tf.boolean_mask(y_pred, index)

    epsilon = 0.0001

    rel_error = tf.subtract(tf.divide((y_pred + epsilon), (y_true + epsilon)), 1)
    # rel_error = tf.divide((y_true + epsilon), (y_pred + epsilon)) - 1

    rel_error = y_pred - y_true
    loss = tf.reduce_mean(tf.square(rel_error))

    return loss


def MSLogError_safe(outputs_wt = [], if_rel_err = []):
    def custom_loss(y_true, y_pred):

        n_feature = np.shape(y_true)[1]
        dummy_y = tf.math.scalar_mul(0.0, y_true)

        if np.size(outputs_wt):
            outputs_wt_tf = outputs_wt.reshape((1, n_feature)) + dummy_y

            y_true = tf.math.multiply(y_true, outputs_wt_tf)
            y_pred = tf.math.multiply(y_pred, outputs_wt_tf)

        index = ~tf.math.is_nan(y_true)
        y_true = tf.boolean_mask(y_true, index)
        y_pred = tf.boolean_mask(y_pred, index)

        # y_true_max = tf.reduce_max(y_true, axis=0)
        # y_true_min = tf.reduce_min(y_true, axis=0)

        # y_true_mean = tf.reduce_mean(y_true, axis=0)
        # y_true_std = tf.math.reduce_std(y_true, axis=0)
        # y_true_min = y_true_mean - 2.0*y_true_std
        # y_true_max = y_true_mean + 2.0*y_true_std
        #
        # y_true = (y_true - y_true_min)/(y_true_max - y_true_min)
        # y_pred = (y_pred - y_true_min)/(y_true_max - y_true_min)

        y_dif = tf.math.abs(y_true - y_pred)

        cut_off = 0.0
        y_dif = tf.nn.relu(y_dif-cut_off)

        if np.size(if_rel_err):

            epsilon = 0.0001
            y_log_dif = 10 * tf.math.abs(tf.math.log(y_true + epsilon) - tf.math.log(y_pred + epsilon))

            # cut_off_log = 0.0
            # y_log_dif = tf.nn.relu(y_log_dif-cut_off_log)

            if_rel_err_tf = if_rel_err.reshape((1, n_feature)) + dummy_y
            if_rel_err_tf = tf.boolean_mask(if_rel_err_tf, index)

            y_log_dif = tf.math.multiply(y_log_dif, if_rel_err_tf)

            loss = tf.reduce_mean(1.0*tf.square(y_log_dif) +
                                  tf.square(y_dif))
        else:
            loss = tf.reduce_mean(tf.square(y_dif))

        # 5 overfit small Ca, misses large CaO

        # loss = tf.reduce_mean(tf.square(y_dif))

        return loss
    return custom_loss


def safe_mse_loss(y_true,y_pred):

    # is_nans = tf.math.logical_or(tf.math.is_nan(y_true), tf.math.is_nan(y_pred))
    #
    # per_instance = tf.where(is_nans,
    #                         tf.zeros_like(y_true, dtype=tf.float32),
    #                         tf.square(tf.subtract(y_pred, y_true)))

    index = ~tf.math.is_nan(y_true)
    y_true = tf.boolean_mask(y_true, index)
    y_pred = tf.boolean_mask(y_pred, index)
    return tf.reduce_mean((y_true - y_pred) ** 2)

    # return tf.reduce_mean(per_instance, axis=0)

