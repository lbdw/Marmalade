import matplotlib.pyplot as plt
import numpy as np

def plot_convergence_and_difference(data, model, history, select_mode_phases, select_target_run):

    x_train, y_train, x_test, y_test, X_target_run, Y_target_run = data
    # Check convergence
    # plt.plot(history.history['loss'])
    plt.plot(history.history['root_mean_squared_error'], color='blue', label='train_rmse')
    plt.plot(history.history['val_root_mean_squared_error'], color='burlywood', label='val_rmse')
    # plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    # plot prediction vs. expected
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    target_run_predictions = model.predict(X_target_run)

    i_fig = 0
    for ii in range(len(select_mode_phases)):
        plt.figure(i_fig)
        plt.scatter(y_train[:, ii], train_predictions[:, ii], c='blue')
        plt.title(select_mode_phases[ii] + " mode")
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        max_val = max(y_train)
        plt.xlim(0,max_val)
        plt.ylim(0,max_val)
        plt.plot([0, 1], [0, 1])
        plt.show()
        i_fig = i_fig+1

        plt.figure(i_fig)
        plt.scatter(y_test[:, ii], test_predictions[:, ii], c='burlywood')
        plt.title(select_mode_phases[ii] + " test")
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        # max_val = max(y_test)
        # plt.xlim(0,max_val)
        # plt.ylim(0,max_val)
        plt.plot([0, 1], [0, 1])
        plt.show()
        i_fig = i_fig+1

        plt.figure(i_fig)
        plt.scatter(Y_target_run[:, ii], target_run_predictions[:, ii], c='forestgreen')
        plt.title(select_mode_phases[ii] + " " + select_target_run[0] + " " + select_target_run[1])
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        max_val = max(Y_target_run)
        plt.xlim(0,max_val)
        plt.ylim(0,max_val)
        plt.plot([0, 1], [0, 1])
        plt.show()
        i_fig = i_fig+1

def plot_convergence(history):

    # Check convergence
    # plt.plot(history.history['loss'])
    plt.plot(history.history['loss'], color='blue', label='train_loss')
    plt.plot(history.history['val_loss'], color='burlywood', label='val_loss')
    # plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

def plot_difference(data, model, y_names, select_target_run, i_cavier = -1):

    x_train, y_train, x_test, y_test, X_target_run, Y_target_run = data
    # Check convergence
    # plt.plot(history.history['loss'])

    # plot prediction vs. expected
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    target_run_predictions = model.predict(X_target_run)

    i_fig = 0
    for ii in range(len(y_names)):
        plt.figure(i_fig)
        plt.scatter(y_train[:, ii], train_predictions[:, ii], c='blue')
        if i_cavier>=0:
            # plt.title(y_names[ii] + " train" + str(i_cavier))
            plt.title(y_names[ii])
        else:
            plt.title(y_names[ii] + " train")
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        max_val = max(y_train[:, ii])
        # plt.xlim(-0.01,max_val)
        # plt.ylim(-0.01,max_val)
        plt.plot([0, max_val], [0, max_val], c='black')
        # plt.savefig('./check_fig/'+ y_names[ii] + " mode" + str(i_cavier) +'.png')
        filename = y_names[ii].replace(":", "_")
        plt.savefig('./check_fig/'+ filename + '.png')
        plt.show()
        i_fig = i_fig+1

        if i_cavier<0:
            plt.figure(i_fig)
            plt.scatter(y_test[:, ii], test_predictions[:, ii], c='burlywood')
            plt.title(y_names[ii] + " test")
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.axis('equal')
            plt.axis('square')
            # max_val = max(y_test[:, ii])
            # plt.xlim(0,max_val)
            # plt.ylim(0,max_val)
            plt.plot([0, 1], [0, 1])
            plt.show()
            i_fig = i_fig+1

            plt.figure(i_fig)
            plt.scatter(Y_target_run[:, ii], target_run_predictions[:, ii], c='forestgreen')
            plt.title(y_names[ii] + " " + select_target_run[0] + " " + select_target_run[1])
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.axis('equal')
            plt.axis('square')
            max_val = max(Y_target_run[:, ii])
            # plt.xlim(0,max_val)
            # plt.ylim(0,max_val)
            plt.plot([0, max_val], [0, max_val])
            plt.show()
            i_fig = i_fig+1

def make_checkpoint_path(y_names, model_type):
    checkpoint_path = './checkpoints/' + model_type

    for ii in range(min(len(y_names), 8)):
        checkpoint_path = checkpoint_path + '_' + y_names[ii]
    checkpoint_path = checkpoint_path + '_checkpoint'

    return checkpoint_path

def plot_ee(var_name,labels,Y_Pred_indices, plot_indices_lists, legend_config = {}):
    x_var_name,y_var_name = var_name
    Y, Y_pred, run_original_indices = Y_Pred_indices
    plot_x = Y[:, labels.index(x_var_name)]
    plot_y = Y[:, labels.index(y_var_name)]

    plot_x_pred = Y_pred[:, labels.index(x_var_name)]
    plot_y_pred = Y_pred[:, labels.index(y_var_name)]

    plt.figure(figsize=(8, 6), dpi=300)

    color_list = ('b', 'r', 'g', 'c', 'm', 'y', 'k', 'Orange', 'Lime', 'Violet', 'Gold')
    ii = 0
    for plot_indices_label in plot_indices_lists:
        idx = []
        plot_indices, label_run = plot_indices_label

        if not legend_config:
            label_run = ''

        for index_val in plot_indices:
            index_all_variants = np.where(run_original_indices == index_val)
            if np.size(index_all_variants):
                idx.append(index_all_variants[0][0])
        if idx:
            plt.plot(plot_x[idx], plot_y[idx], 'o', color=color_list[ii], label = label_run)
            plt.plot(plot_x_pred[idx], plot_y_pred[idx], '+-', color=color_list[ii], label = '')

        ii = ii + 1

    plt.xlabel(x_var_name)
    plt.ylabel(y_var_name)

    if legend_config:
        plt.legend(bbox_to_anchor=legend_config["bbox_to_anchor"], loc=legend_config["loc"])


    plt.show()