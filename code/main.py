import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from multiprocessing import Pool

from config_deep_rock import config_device
from model_def import build_model, build_linear_model, build_ann_model, build_xann_model, norm_loss, safe_mse_loss
from import_project_data import import_simple_data, import_lepr_expanded_data, format_lepr_data, get_XY_data_phase_mode, get_XY_data_phase_composition
from utilities import plot_convergence, plot_difference, make_checkpoint_path

config_device()

import magma_differentiation

# Import data
print("Import data")
# x_train, y_train = import_simple_data()

data_version = "02_14_2021b"
data_version = "02_14_2021i"
# data_version = "02_13_2021b"
# data_version = "02_13_2021i"
# data_version = "02_12_2021b"
# data_version = "02_12_2021i"
# data_version = "02_09_2021b"
# data_version = "02_09_2021i"
# data_version = "02_06_2021b"
# data_version = "02_06_2021i"
# data_version = "02_01_2021i"
# data_version = "02_01_2021s"
# data_version = "02_01_2021b"
# data_version = "01_29_2021i"
lepr_synthetic_data_folder = "C:\\Users\\boda\\OneDrive\\project\\FAN\\LEPR_Synthetic_Big_Data\\"

lepr_info_filename = lepr_synthetic_data_folder + "LEPR_BIG_" + data_version + ".xlsx"
input_filename = lepr_synthetic_data_folder + "LEPR_BIG_Input_" + data_version + ".txt"
mode_filename = lepr_synthetic_data_folder + "LEPR_BIG_Phase_mode_" + data_version + ".txt"
phase_composition_filename = lepr_synthetic_data_folder + "LEPR_BIG_Phase_compositions_" + data_version + ".txt"
norm_data_filename = lepr_synthetic_data_folder + "LEPR_norm_data.xlsx"

read_lepr_data = import_lepr_expanded_data(lepr_info_filename,
                                           input_filename,
                                           mode_filename,
                                           phase_composition_filename,
                                           norm_data_filename,
                                           summary_output = 1)

# select_input_features = ['SiO2','TiO2','Al2O3','Cr2O3','FeO','MnO','MgO','CaO','Na2O','K2O', 'P(GPa)', 'T(C)', 'H2O','Original indices']  #  for mode

# for phase mode
select_input_features = ['SiO2','TiO2','Al2O3','Cr2O3','FeO','MnO','MgO','CaO','Na2O','K2O', 'P(GPa)', 'T(C)', 'H2O', 'Original indices']

#  for phase comp 'log fO2'
# select_input_features = ['SiO2','TiO2','Al2O3','FeO','MgO','CaO','Na2O','K2O', 'P(GPa)', 'T(C)', 'log fO2', 'H2O', 'Original indices']

# select_mode_phases = ['Liquid', 'Clinopyroxene','Olivine','Garnet','Plagioclase','Spinel','Amphibole',
#                       'Orthopyroxene','Ilmenite','Titanite','Rutile','Oxide-generic','Quartz','Potassium feldspar','Biotite']

# select_mode_phases_list = ('Liquid', 'Clinopyroxene','Olivine','Orthopyroxene','Garnet','Plagioclase','Spinel','Amphibole')
# select_mode_phases_list = ('Ilmenite',)
# select_mode_phases_list = ('Plagioclase',)
# select_mode_phases_list = (['Liquid', 'Garnet','Plagioclase','Amphibole'],)
# select_mode_phases_list = (['Liquid', 'Clinopyroxene','Olivine','Garnet','Plagioclase','Spinel','Amphibole',
#                       'Orthopyroxene','Ilmenite','Titanite','Rutile','Oxide-generic','Quartz','Potassium feldspar','Biotite'],)


select_mode_phases_list = (['Liquid', 'Clinopyroxene', 'Olivine', 'Garnet', 'Plagioclase', 'Amphibole', 'Orthopyroxene'],)  # 64x3 ANN

# select_mode_phases_list = (['Liquid'],) # 64x3 XANN


# select_mode_phases_list = (['Spinel','Quartz','Potassium feldspar','Biotite'],)
# select_mode_phases_list = (['Spinel','Ilmenite','Titanite','Rutile','Oxide-generic','Potassium feldspar','Biotite'],)

# select_mode_phases_list = (['Spinel'],)

for select_mode_phase in select_mode_phases_list:
    select_mode_phases = select_mode_phase

    select_feature = (select_mode_phases, select_input_features)
    select_target_run = ('Ulmer', '2018', 0)
    select_target_run = ('Alonso-Perez', '2009', 0)
    # select_target_run = ('Gaetani', '1998',0) # not consistent with decreasing grt during isobaric melting

    # X, Y, X_target_run, Y_target_run, input_stats = format_lepr_data(read_lepr_data, select_feature, select_target_run)

    # X, Y, X_target_run, Y_target_run, input_stats = get_XY_data_phase_mode(read_lepr_data, select_feature, select_target_run)

    # for mode
    X, Y, X_target_run, Y_target_run, input_stats = format_lepr_data(read_lepr_data, select_feature, select_target_run, y_range = [0, 1])

    # for phase comp
    # fill_fO2 = 1
    # phase_compositions_columns_select, X, Y, X_target_run, Y_target_run, input_stats = get_XY_data_phase_composition(read_lepr_data, select_feature, select_target_run, fill_fO2)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=0)

    x_train = X
    y_train = Y

    # x_test = X
    # y_test = Y

    train_original_indices = x_train[:, -1]
    test_original_indices = x_test[:, -1]
    target_run_original_indices = X_target_run[:, -1]

    x_train = np.delete(x_train, -1, 1)
    x_test = np.delete(x_test, -1, 1)
    X_target_run = np.delete(X_target_run, -1, 1)

    train_len = np.shape(x_train)[0]

    data = (x_train, y_train, x_test, y_test, X_target_run, Y_target_run)

    print("Build model")
    #################################################################################
    # Sequential neurual network model
    print("Neural network model")

    for i_cavier in range(1):

        # for mode
        model = build_ann_model(x_test.shape[1], y_test.shape[1])

        # for phase comp
        # if_melt_water = 1
        # model = build_xann_model(x_test.shape[1], y_test.shape[1], if_melt_water)

        # make the path to save or load weights
        checkpoint_path = make_checkpoint_path(select_mode_phases, 'ann')
        # checkpoint_path = make_checkpoint_path(select_mode_phases, 'xann')

        # load weights
        print("Load weights")

        # model.load_weights(checkpoint_path + '_updated')
        model.load_weights(checkpoint_path)

        # for mode
        # plot_difference(data, model, select_mode_phases, select_target_run,0)

        # for compositions
        # plot_difference(data, model, phase_compositions_columns_select, select_target_run,0)

        model.summary()
        print('Data set length: ' + str(train_len))

        # Configure Optimizer and Train
        model.compile(
            # optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
            # optimizer=tf.keras.optimizers.SGD(lr=10000000),
            # optimizer=tf.keras.optimizers.Adam(lr=0.00001),
            loss=safe_mse_loss,
            # loss=norm_loss,
            # loss='mean_squared_logarithmic_error',
            # metrics = safe_mse_loss
            # metrics=tf.keras.metrics.RootMeanSquaredError()
        )

        print("Fit model on training data")
        # history = model.fit(
        #                     # X_target_run, Y_target_run,
        #                     x_train, y_train,
        #                     batch_size=int(train_len/1),
        #                     # batch_size = 200,
        #                     epochs=1000,
        #                     validation_data = (x_test, y_test),
        #                     # validation_data = (X_target_run, Y_target_run),
        #                     verbose=1)
        #
        # plot_convergence(history)

        # for mode
        plot_difference(data, model, select_mode_phases, select_target_run,0)

        # for compositions
        # plot_difference(data, model, phase_compositions_columns_select, select_target_run,0)


        ##############
        # labels = phase_compositions_columns_select
        labels = select_mode_phases

        train_predictions = model.predict(x_train)
        test_predictions = model.predict(x_test)
        target_run_predictions = model.predict(X_target_run)
        #
        i_fig = 0
        # for ii in range(len(labels)):
        #     plt.figure(i_fig)
        #     plt.scatter(y_train[:, ii], train_predictions[:, ii], c='blue')
        #     if i_cavier >= 0:
        #         plt.title(labels[ii] + " train" + str(i_cavier))
        #     else:
        #         plt.title(labels[ii] + " train")
        #     plt.xlabel('True Values')
        #     plt.ylabel('Predictions')
        #     plt.axis('equal')
        #     plt.axis('square')
        #     max_val = max(y_train[:, ii])
        #     # plt.xlim(-0.01,max_val)
        #     # plt.ylim(-0.01,max_val)
        #     plt.plot([0, max_val], [0, max_val], c='black')
        #     plt.show()
        #     i_fig = i_fig + 1
        #
        # # # Save the weights
        # model.save_weights(checkpoint_path)
        # model.save_weights(checkpoint_path + '_updated')

        # model.save_weights(checkpoint_path +'_' + str(i_cavier))
        # model.save_weights(checkpoint_path + '_updated' +'_' + str(i_cavier))

#################################################################################
# Linear model
# model = build_linear_model(x_test.shape[1], y_test.shape[1])
# model.summary()
#
# # Configure Optimizer and Train
# model.compile(
#     # optimizer=tf.keras.optimizers.RMSprop(lr=0.000000001),
#     optimizer=tf.keras.optimizers.SGD(),
#     # optimizer=tf.keras.optimizers.Adam(lr=0.00001),
#     loss='mse',
#     # loss=norm_loss,
#     # loss='mean_squared_logarithmic_error',
#     metrics=tf.keras.metrics.RootMeanSquaredError())
#
# print("Fit model on training data")
# history = model.fit(
#     # X_target_run, Y_target_run,
#     x_train, y_train,
#                     batch_size=int(train_len/10),
#                     # batch_size = 200,
#                     epochs=1000,
#                     validation_data = (x_test, y_test),
#                     # validation_data = (X_target_run, Y_target_run),
#                     verbose=1)
#
# plot_convergence(history)
# plot_difference(data, model, select_mode_phases, select_target_run, 0)


print('Data set length ' + str(train_len))

print("Chill.")

# Show model parameters
# print("Model weights:")
# for layer in model.layers:
#     weights = layer.get_weights()
#     print(weights)
