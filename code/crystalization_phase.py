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


# Import data
print("Import data")
# x_train, y_train = import_simple_data()

data_version = "03_15_2021i"
# data_version = "03_14_2021b"
# data_version = "02_14_2021b"
# data_version = "02_14_2021i"
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


# for phase mode
select_input_features = ['SiO2','TiO2','Al2O3','Cr2O3','FeO','MnO','MgO','CaO','Na2O','K2O', 'P(GPa)', 'T(C)', 'H2O', 'Original indices']
select_mode_phases_list = (['Liquid', 'Clinopyroxene', 'Olivine', 'Garnet', 'Plagioclase', 'Amphibole', 'Orthopyroxene'],)  # 64x3 ANN

#  for phase comp 'log fO2'
# select_input_features = ['SiO2','TiO2','Al2O3','FeO','MgO','CaO','Na2O','K2O', 'P(GPa)', 'T(C)', 'log fO2', 'H2O', 'Original indices']
# select_mode_phases_list = (['Liquid'],) # 64x3 XANN


for select_mode_phase in select_mode_phases_list:
    select_mode_phases = select_mode_phase

    select_feature = (select_mode_phases, select_input_features)
    select_target_run = ('Ulmer', '2018', 0)
    select_target_run = ('Alonso-Perez', '2009', 0)
    # select_target_run = ('Gaetani', '1998',0) # not consistent with decreasing grt during isobaric melting

    # for mode
    X, Y, X_target_run, Y_target_run, input_stats = format_lepr_data(read_lepr_data, select_feature, select_target_run, y_range = [0, 1])


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

        checkpoint_path = make_checkpoint_path(select_mode_phases, 'ann')

        # load weights
        print("Load weights")

        model.load_weights(checkpoint_path)

        # for mode
        # plot_difference(data, model, select_mode_phases, select_target_run,0)


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
        # plot_difference(data, model, select_mode_phases, select_target_run,0)

        ##############
        # labels = phase_compositions_columns_select
        labels = select_mode_phases

        train_predictions = model.predict(x_train)
        test_predictions = model.predict(x_test)
        target_run_predictions = model.predict(X_target_run)
        #
        i_fig = 0
        for ii in range(len(labels)):
            plt.figure(i_fig)
            plt.scatter(y_train[:, ii], train_predictions[:, ii], c='blue')
            if i_cavier >= 0:
                plt.title(labels[ii] + " train" + str(i_cavier))
            else:
                plt.title(labels[ii] + " train")
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.axis('equal')
            plt.axis('square')
            max_val = max(y_train[:, ii])
            # plt.xlim(-0.01,max_val)
            # plt.ylim(-0.01,max_val)
            plt.plot([0, max_val], [0, max_val], c='black')
            plt.show()
            i_fig = i_fig + 1

        ##############
        labels = select_mode_phases

        train_predictions = model.predict(x_train)
        test_predictions = model.predict(x_test)
        target_run_predictions = model.predict(X_target_run)

        # ########### Ulmer et al. (2018) 1GPa ##############################################################

        # range(9564, 9579): Fractional crystallization RC158c AuPd capsules (FC Mb AuPd) 1GPa
        # range(9596, 9611): Equilibrium crystallization RC158c graphite–Pt capsules (EQ Mb Pt–C)

        # range(9579, 9590): Equilibrium crystallization RC158c AuPd capsules (EQ Mb AuPd)

        ## range(9547, 9564): Fractional crystallization 85-44 AuPd capsules (FC ba AuPd)

        Ulmer_2018_FC = (range(9564, 9579), 'Ulmer et al. (2018) FC, 1GPa')
        Ulmer_2018_EC = (range(9579, 9590), 'Ulmer et al. (2018) EC, 1GPa')
        Ulmer_2018_EC_r = (range(9596, 9611), 'Ulmer et al. (2018) EC, 1GPa, reduced')

        # ########### Alonso-Perez et al. (2009) ##############################################################

        # (54963, 54964, 54965, 54972): Equilibrium crystallization: 1.2 GPa, 4%H2O
        # range(54983, 54987): Equilibrium crystallization: 0.8 GPa, 4%H2O

        Alonso_2009_EC = ((54963, 54964, 54965, 54972), 'Alonso-Perez et al. (2009) FC, 1.2GPa')

        # ########### Müntener, O., Kelemen, P.B., Grove, T.L. (2001) ##############################################################

        # range(2255, 2268) sorted (2259,2265,2258,2264,2257,2263,2267,2256,2262,2266,2261,2255,2260): Equilibrium crystallization: 1.2 GPa basalt 85-44
        # range(2270, 2276): Equilibrium crystallization: 1.2 GPa High Mg# andesite 85-41c

        Muntener_2001_EC_b = ((2259, 2265, 2258, 2264, 2257, 2263, 2267, 2256, 2262, 2266, 2261, 2255, 2260),
                              'Müntener et al. (2001) EC, BA, 1.2GPa')
        Muntener_2001_EC_a = (range(2270, 2276), 'Müntener et al. (2001) EC, Hi-Mg A, 1.2GPa')

        # ########### Nandedkar, R. H., P. Ulmer and O. Müntener (2014) ##############################################################

        Nandedkar_2014_FC = (range(9530, 9547), 'Nandedkar et al. (2014) FC, 0.7GPa')

        # ########### Blatter, Sisson, Hankins (2013) ##############################################################

        Blatter_2013_EC_09GPa = (range(9611, 9622), 'Blatter et al. (2012) EC, 0.9GPa')
        Blatter_2013_EC_07GPa = (range(9622, 9630), 'Blatter et al. (2012) EC, 0.7GPa')
        Blatter_2013_EC_04GPa = (range(9630, 9636), 'Blatter et al. (2012) EC, 0.4GPa')

        # ########### Sisson, T.W., Grove, T.L. (1993) ##############################################################

        Sisson_Grove_1993_HT = (range(20029, 20034), 'Sisson&Grove (1993) EC, Hi-T 0.2GPa, Sat')
        Sisson_Grove_1993 = (range(20039, 20046), 'Sisson&Grove (1993) EC, 0.2GPa, Sat')

        plot_indices_lists = (
            Ulmer_2018_EC, Ulmer_2018_FC,
            Alonso_2009_EC,
            Muntener_2001_EC_b, Muntener_2001_EC_a,
            Nandedkar_2014_FC,
            Blatter_2013_EC_09GPa,
            Sisson_Grove_1993_HT, Sisson_Grove_1993)

        # Y_Pred_indices = (Y_target_run, target_run_predictions, target_run_original_indices)
        Y_Pred_indices = (y_train, train_predictions, train_original_indices)

        x_var_name = 'Liquid:MgO'
        y_var_name = 'Liquid:Al2O3'
        legend_config = {"bbox_to_anchor": (0.5, 0.6), "loc": 'lower left'}
        var_name = (x_var_name, y_var_name)
        # plot_Tm(var_name, labels, Y_Pred_indices, plot_indices_lists, legend_config)

        # # # Save the weights
        # model.save_weights(checkpoint_path)
        # model.save_weights(checkpoint_path + '_updated')

        # model.save_weights(checkpoint_path +'_' + str(i_cavier))
        # model.save_weights(checkpoint_path + '_updated' +'_' + str(i_cavier))

print('Data set length ' + str(train_len))

print("Chill.")

# Show model parameters
# print("Model weights:")
# for layer in model.layers:
#     weights = layer.get_weights()
#     print(weights)
