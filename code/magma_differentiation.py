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
from model_def import build_model, build_linear_model, build_ann_model, build_xann_model, norm_loss, safe_mse_loss, MSLogError_safe, build_x_model_SiTiAlFeMgCaNaKH, build_x_model_SiTiAlFeMgCaNaKH_reg, build_xann_model_fine
from import_project_data import import_simple_data, import_lepr_expanded_data, format_lepr_data, get_XY_data_phase_mode, get_XY_data_phase_composition
from utilities import plot_convergence, plot_difference, make_checkpoint_path, plot_ee

config_device()

# Import data
print("Import data")
# x_train, y_train = import_simple_data()


data_version = "03_16_2021b"
# data_version = "03_15_2021b"
# data_version = "03_14_2021b"
# data_version = "03_08_2021b"
# data_version = "03_08_2021i"
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

# select_input_features = ['SiO2','TiO2','Al2O3','Cr2O3','FeO','MnO','MgO','CaO','Na2O','K2O', 'P(GPa)', 'T(C)', 'H2O','Original indices']  #  for mode

# for phase mode
# select_input_features = ['SiO2','TiO2','Al2O3','Cr2O3','FeO','MnO','MgO','CaO','Na2O','K2O', 'P(GPa)', 'T(C)', 'H2O', 'Original indices']
# select_mode_phases_list = (['Liquid', 'Clinopyroxene', 'Olivine', 'Garnet', 'Plagioclase', 'Amphibole', 'Orthopyroxene'],)  # 64x3 ANN

#  for phase comp 'log fO2'
# select_input_features = ['SiO2','Al2O3','FeO','MgO','CaO',
#                          'TiO2','Na2O','K2O',
#                          'P(GPa)', 'T(C)', 'log fO2', 'H2O', 'Original indices']

select_input_features = ['SiO2','TiO2','Al2O3','FeO','MgO','CaO','Na2O','K2O','P(GPa)', 'T(C)', 'log fO2', 'H2O', 'Original indices']
select_mode_phases_list = (['Liquid'],) # 64x3 XANN


for select_mode_phase in select_mode_phases_list:
    select_mode_phases = select_mode_phase

    select_feature = (select_mode_phases, select_input_features)
    select_target_run = ('Ulmer', '2018', 0)
    # select_target_run = ('Alonso-Perez', '2009', 0)
    # select_target_run = ('M??ntener', '2001', 1)
    # select_target_run = ('Gaetani', '1998',0) # not consistent with decreasing grt during isobaric melting

    # for mode
    # X, Y, X_target_run, Y_target_run, input_stats = format_lepr_data(read_lepr_data, select_feature, select_target_run, y_range = [0, 1])

    # for phase comp
    fill_fO2 = 1
    phase_compositions_columns_select, X, Y, X_target_run, Y_target_run, input_stats, phase_compositions_columns_select_err = get_XY_data_phase_composition(read_lepr_data, select_feature, select_target_run, fill_fO2)

    phase_compositions_columns_select_wt = 1/phase_compositions_columns_select_err

    phase_compositions_columns_if_rel_err = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    phase_compositions_columns_if_rel_err[1] = 1
    phase_compositions_columns_if_rel_err[6] = 1
    phase_compositions_columns_if_rel_err[7] = 1

    # phase_compositions_columns_if_rel_err[3] = 1
    # phase_compositions_columns_if_rel_err[4] = 1
    # phase_compositions_columns_if_rel_err[5] = 1

    # 'SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'H2O'
    #   0        1       2       3      4       5      6      7      8

    #
    phase_compositions_columns_select_wt[0] = 0

    # phase_compositions_columns_select_wt[1] = 0
    # # #
    phase_compositions_columns_select_wt[2] = 0
    # # #
    phase_compositions_columns_select_wt[3] = 0
    # # # #
    phase_compositions_columns_select_wt[4] = 0
    # #
    phase_compositions_columns_select_wt[5] = 0
    #
    # phase_compositions_columns_select_wt[6] = 0
    # #
    # phase_compositions_columns_select_wt[7] = 0
    # #
    # phase_compositions_columns_select_wt[8] = 0

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
        # model = build_ann_model(x_test.shape[1], y_test.shape[1])

        # checkpoint_path = make_checkpoint_path(select_mode_phases, 'ann')

        # for phase comp
        if_melt_water = 1
        model = build_xann_model(x_test.shape[1])
        # model = build_x_model_SiTiAlFeMgCaNaKH(x_test.shape[1], y_test.shape[1])
        # model = build_x_model_SiTiAlFeMgCaNaKH_reg(x_test.shape[1], y_test.shape[1])
        # model = build_xann_model_fine(x_test.shape[1], y_test.shape[1])  # SiAlFeMgCa + TiNaK + H20

        checkpoint_path = make_checkpoint_path(select_mode_phases, 'xann')

        # load weights

        # model.load_weights(checkpoint_path + '_updated')
        model.load_weights(checkpoint_path)

        # for mode
        # plot_difference(data, model, select_mode_phases, select_target_run,0)

        # for compositions
        # plot_difference(data, model, phase_compositions_columns_select, select_target_run,0)

        model.summary()
        print('Data set length: ' + str(train_len))

        ##
        # train_predictions = model.predict(x_train)
        #
        # outputs_wt = phase_compositions_columns_select_wt
        # if_rel_err = phase_compositions_columns_if_rel_err

        # def custom_loss(y_true, y_pred):
        #
        #     n_feature = np.shape(y_true)[1]
        #     dummy_y = tf.math.scalar_mul(0.0, y_true)
        #
        #     if np.size(outputs_wt):
        #         outputs_wt_tf = outputs_wt.reshape((1, n_feature)) + dummy_y
        #
        #         y_true = tf.math.multiply(y_true, outputs_wt_tf)
        #         y_pred = tf.math.multiply(y_pred, outputs_wt_tf)
        #
        #     index = ~tf.math.is_nan(y_true)
        #     y_true = tf.boolean_mask(y_true, index)
        #     y_pred = tf.boolean_mask(y_pred, index)
        #
        #     # y_true_max = tf.reduce_max(y_true, axis=0)
        #     # y_true_min = tf.reduce_min(y_true, axis=0)
        #
        #     # y_true_mean = tf.reduce_mean(y_true, axis=0)
        #     # y_true_std = tf.math.reduce_std(y_true, axis=0)
        #     # y_true_min = y_true_mean - 2.0*y_true_std
        #     # y_true_max = y_true_mean + 2.0*y_true_std
        #     #
        #     # y_true = (y_true - y_true_min)/(y_true_max - y_true_min)
        #     # y_pred = (y_pred - y_true_min)/(y_true_max - y_true_min)
        #
        #     y_dif = tf.math.abs(y_true - y_pred)
        #
        #     cut_off = 0.0
        #     y_dif = tf.nn.relu(y_dif-cut_off)
        #
        #     if np.size(if_rel_err):
        #
        #         epsilon = 0.0001
        #         y_log_dif = 10 * tf.math.abs(tf.math.log(y_true + epsilon) - tf.math.log(y_pred + epsilon))
        #
        #         # cut_off_log = 0.0
        #         # y_log_dif = tf.nn.relu(y_log_dif-cut_off_log)
        #
        #         if_rel_err_tf = if_rel_err.reshape((1, n_feature)) + dummy_y
        #         y_log_dif = tf.math.multiply(y_log_dif, if_rel_err_tf)
        #
        #         loss = tf.reduce_mean(1.0*tf.square(y_log_dif) +
        #                               tf.square(y_dif))
        #     else:
        #         loss = tf.reduce_mean(tf.square(y_dif))
        #
        #     # 5 overfit small Ca, misses large CaO
        #
        #     # loss = tf.reduce_mean(tf.square(y_dif))
        #
        #     return loss
        #
        # custom_loss(y_train, train_predictions)

        ## Configure Optimizer and Train
        model.compile(
            # optimizer=tf.keras.optimizers.RMSprop(lr=1e-9),
            # optimizer=tf.keras.optimizers.SGD(lr=1e-9),
            optimizer=tf.keras.optimizers.Adam(),
            # loss=safe_mse_loss,
            # loss=norm_loss,
            loss=MSLogError_safe(outputs_wt = phase_compositions_columns_select_wt, if_rel_err = phase_compositions_columns_if_rel_err),
            # loss='mean_squared_logarithmic_error',
            # metrics = safe_mse_loss
            # metrics=tf.keras.metrics.RootMeanSquaredError()
        )

        print("Fit model on training data")
        history = model.fit(
                            # X_target_run, Y_target_run,
                            x_train, y_train,
                            batch_size=int(train_len/1),
                            # batch_size = 200,
                            epochs=1000,
                            validation_data = (x_test, y_test),
                            # validation_data = (X_target_run, Y_target_run),
                            verbose=1)

        plot_convergence(history)

        plot_difference(data, model, phase_compositions_columns_select, select_target_run,0)

        ##############
        labels = phase_compositions_columns_select
        # labels = select_mode_phases

        train_predictions = model.predict(x_train)
        test_predictions = model.predict(x_test)
        target_run_predictions = model.predict(X_target_run)

        # ########### Ulmer et al. (2018) 1GPa ##############################################################

        # range(9564, 9579): Fractional crystallization RC158c AuPd capsules (FC Mb AuPd) 1GPa
        # range(9596, 9611): Equilibrium crystallization RC158c graphite???Pt capsules (EQ Mb Pt???C)

        # range(9579, 9590): Equilibrium crystallization RC158c AuPd capsules (EQ Mb AuPd)

        ## range(9547, 9564): Fractional crystallization 85-44 AuPd capsules (FC ba AuPd)

        Ulmer_2018_FC = (range(9564, 9579), 'Ulmer et al. (2018) FC, 1GPa')
        Ulmer_2018_EC = (range(9579, 9590), 'Ulmer et al. (2018) EC, 1GPa')
        Ulmer_2018_EC_r = (range(9596, 9611), 'Ulmer et al. (2018) EC, 1GPa, reduced')

        # ########### Alonso-Perez et al. (2009) ##############################################################

        # (54963, 54964, 54965, 54972): Equilibrium crystallization: 1.2 GPa, 4%H2O
        # range(54983, 54987): Equilibrium crystallization: 0.8 GPa, 4%H2O

        Alonso_2009_EC = ((54963, 54964, 54965, 54972), 'Alonso-Perez et al. (2009) FC, 1.2GPa')

        # ########### M??ntener, O., Kelemen, P.B., Grove, T.L. (2001) ##############################################################

        # range(2255, 2268) sorted (2259,2265,2258,2264,2257,2263,2267,2256,2262,2266,2261,2255,2260): Equilibrium crystallization: 1.2 GPa basalt 85-44
        # range(2270, 2276): Equilibrium crystallization: 1.2 GPa High Mg# andesite 85-41c

        Muntener_2001_EC_b = ((2259,2265,2258,2264,2257,2263,2267,2256,2262,2266,2261,2255,2260), 'M??ntener et al. (2001) EC, BA, 1.2GPa')
        Muntener_2001_EC_a = (range(2270, 2276), 'M??ntener et al. (2001) EC, Hi-Mg A, 1.2GPa')

        # ########### Nandedkar, R. H., P. Ulmer and O. M??ntener (2014) ##############################################################

        Nandedkar_2014_FC = (range(9530, 9547), 'Nandedkar et al. (2014) FC, 0.7GPa')

        # ########### Blatter, Sisson, Hankins (2013) ##############################################################

        Blatter_2013_EC_09GPa = (range(9611, 9622), 'Blatter et al. (2012) EC, 0.9GPa')
        Blatter_2013_EC_07GPa = (range(9622, 9630), 'Blatter et al. (2012) EC, 0.7GPa')
        Blatter_2013_EC_04GPa = (range(9630, 9636), 'Blatter et al. (2012) EC, 0.4GPa')


        # ########### Villiger, S., Ulmer, P., Muntener, O., and Thompson, A.B. (2004) ##############################################################

        Villiger_et_al_2004_EC = (range(3900, 3908), 'Villiger et al. (2004) EC, 1GPa, Anhyr')
        Villiger_et_al_2004_FC = (range(3908, 3918), 'Villiger et al. (2004) FC, 1GPa, Anhyr')

        # ########### Villiger, S., Ulmer, P., and M??ntener, O. (2007) ##############################################################

        Villiger_et_al_2007_EC = (range(55078, 55082), 'Villiger et al. (2007) EC, 0.7GPa, Anhyr')
        Villiger_et_al_2007_FC = (range(55082, 55090), 'Villiger et al. (2007) FC, 0.7GPa, Anhyr')

        # ########### Sisson, T.W., Grove, T.L. (1993) ##############################################################

        Sisson_Grove_1993_HT = (range(20029, 20034), 'Sisson&Grove (1993) EC, Hi-T 0.2GPa, Sat')
        Sisson_Grove_1993_LT = (range(20039, 20046), 'Sisson&Grove (1993) EC, Lo-T 0.2GPa, Sat')

        Sisson_Grove_1993 = (list(range(20029, 20034)) + list(range(20039, 20046)), 'Sisson&Grove (1993) EC, 0.2GPa, Sat')

        # ########### T. W. Sisson, K. Ratajeski, W. B. Hankins, A. F. Glazner (2005) ##############################################################

        Sisson_et_al_2005_ECo1_07GPa = (range(9653, 9658), 'Sisson et al. (2005) ECo, 0.7GPa')
        Sisson_et_al_2005_ECr1_07GPa = (range(9636, 9643), 'Sisson et al. (2005) ECr, 0.7GPa')

        Sisson_et_al_2005_ECo2_07GPa = (range(9668, 9673), 'Sisson et al. (2005) ECr, 0.7GPa')
        Sisson_et_al_2005_ECr2_07GPa = (range(9658, 9663), 'Sisson et al. (2005) ECo, 0.7GPa')

        Sisson_et_al_2005_ECo3_07GPa = (range(9668, 9673), 'Sisson et al. (2005) ECr, 0.7GPa')
        Sisson_et_al_2005_ECr3_07GPa = (range(9680, 9685), 'Sisson et al. (2005) ECo, 0.7GPa')

        plot_indices_lists = (
                                Ulmer_2018_EC, Ulmer_2018_FC,
                                Alonso_2009_EC,
                                Muntener_2001_EC_a,
                                Nandedkar_2014_FC,
                                # Blatter_2013_EC_09GPa,
                                Sisson_Grove_1993_LT,
                                Villiger_et_al_2004_EC, Villiger_et_al_2004_FC,
                                Villiger_et_al_2007_EC, Villiger_et_al_2007_FC)

        # Y_Pred_indices = (Y_target_run, target_run_predictions, target_run_original_indices)
        Y_Pred_indices = (y_train, train_predictions, train_original_indices)

        x_var_name = 'Liquid:MgO'
        y_var_name = 'Liquid:Al2O3'
        legend_config = {"bbox_to_anchor": (0.5, 0.6), "loc": 'lower left'}
        var_name = (x_var_name, y_var_name)
        plot_ee(var_name, labels, Y_Pred_indices, plot_indices_lists, legend_config)

        x_var_name = 'Liquid:MgO'
        y_var_name = 'Liquid:SiO2'
        var_name = (x_var_name, y_var_name)
        plot_ee(var_name, labels, Y_Pred_indices, plot_indices_lists)

        x_var_name = 'Liquid:MgO'
        y_var_name = 'Liquid:FeO'
        var_name = (x_var_name, y_var_name)
        plot_ee(var_name, labels, Y_Pred_indices, plot_indices_lists)

        x_var_name = 'Liquid:MgO'
        y_var_name = 'Liquid:CaO'
        var_name = (x_var_name, y_var_name)
        plot_ee(var_name, labels, Y_Pred_indices, plot_indices_lists)

        x_var_name = 'Liquid:MgO'
        y_var_name = 'Liquid:H2O'
        var_name = (x_var_name, y_var_name)
        plot_ee(var_name, labels, Y_Pred_indices, plot_indices_lists)

        x_var_name = 'Liquid:MgO'
        y_var_name = 'Liquid:TiO2'
        var_name = (x_var_name, y_var_name)
        plot_ee(var_name, labels, Y_Pred_indices, plot_indices_lists)

        x_var_name = 'Liquid:MgO'
        y_var_name = 'Liquid:Na2O'
        var_name = (x_var_name, y_var_name)
        plot_ee(var_name, labels, Y_Pred_indices, plot_indices_lists)

        x_var_name = 'Liquid:MgO'
        y_var_name = 'Liquid:K2O'
        var_name = (x_var_name, y_var_name)
        plot_ee(var_name, labels, Y_Pred_indices, plot_indices_lists)


        # i_fig = 0
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
        model.save_weights(checkpoint_path)
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
