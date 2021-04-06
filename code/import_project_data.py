import numpy as np
import pandas as pd

# This training set has 4 samples. Each sample has a 2-element input x(i) and 2-element output y(i).
# The true model is linear, i.e. y(i) = w * x(i) + b, where w = [[1, 1], [2, -1]], b = [1, 4]
def import_simple_data():
    x_train = np.array([[1, 2], [4, 5], [7, 8], [3, 10]])
    # y_train = np.array([3, 9, 15, 13]) + 1
    # y_train = np.sum(x_train, axis=1) + 1
    y_train_1 = x_train[:, 0] + x_train[:, 1] + 1
    y_train_2 = 2 * x_train[:, 0] - x_train[:, 1] + 4
    y_train = np.array([y_train_1, y_train_2])
    y_train = y_train.T

    return x_train, y_train

def import_lepr_expanded_data(lepr_info_filename, input_filename, mode_filename, phase_composition_filename, norm_data_filename,summary_output=0):

    references_lookup = pd.read_excel(lepr_info_filename, sheet_name='References')
    input_table = pd.read_table(input_filename, delimiter=',')
    mode_table = pd.read_table(mode_filename, delimiter=',')
    phase_composition_table = pd.read_table(phase_composition_filename, delimiter=',')
    norm_data = pd.read_excel(norm_data_filename, sheet_name='norm_data')
    norm_data = norm_data.set_index('stats')

    # don't have to do this if the columns will be specifically chosen
    # drop the last unnamed column
    # mode_table.drop(mode_table.columns[len(mode_table.columns)-1], axis=1, inplace=True)

    read_lepr_data = (references_lookup, input_table, mode_table, phase_composition_table, norm_data)

    if summary_output>0:
        input_summary = input_table.describe()
        input_summary.to_excel(input_filename.replace('.txt', '_summary.xlsx'))

    return read_lepr_data

def format_lepr_data(read_lepr_data, select_feature, select_target_run,  y_range = []):
    references_lookup, input_table, mode_table, phase_composition_table, norm_data = read_lepr_data
    select_mode_phases, select_input_features = select_feature
    target_run_Author, target_run_Year, if_exclude_target_from_test = select_target_run

    # calculate sum of phase mode
    mode_sum = mode_table.sum(axis=1)
    # valid mode sum row indices
    index_select = mode_sum > 0

    # filter input and mode
    mode_table_select_phase = mode_table.loc[index_select, select_mode_phases]
    input_table_select_feature = input_table.loc[index_select, select_input_features]

    # get rid of samples with nan input values, usually nan occurs in the column of bulk 'H2O'
    row_has_nan = input_table_select_feature.isnull().any(axis=1)
    mode_table_select_phase = mode_table_select_phase[~row_has_nan]
    input_table_select_feature = input_table_select_feature[~row_has_nan]

    # specify the range of y values
    if len(y_range) == 2:
        row_y_constrained = mode_table_select_phase.iloc[:,0].between(y_range[0], y_range[1], inclusive=True)
        mode_table_select_phase = mode_table_select_phase[row_y_constrained]
        input_table_select_feature = input_table_select_feature[row_y_constrained]

    # select Ulmer et al. (2018)
    # Two steps. First, find the reference int he lookup table
    index_lookup_target_run_Author = references_lookup['references'].str.contains(target_run_Author)
    index_lookup_target_run_Year = references_lookup['references'].str.contains(target_run_Year)
    original_index_target_run = references_lookup.loc[
        index_lookup_target_run_Author & index_lookup_target_run_Year, 'index']
    # Second, select index in input_table_select_feature
    index_target_run = input_table_select_feature['Original indices'].isin(original_index_target_run)

    # drop the "Original indices" column in input. because we do not need the indices in the NN input
    input_table_select_feature_drop_indices = input_table_select_feature.drop(columns='Original indices', axis=1)
    # normalization with mean and std

    # input_table_select_feature_stats = input_table_select_feature_drop_indices.describe()

    input_table_select_feature_stats = norm_data[select_input_features]
    input_table_select_feature_stats = input_table_select_feature_stats.drop(columns='Original indices', axis=1)

    input_table_select_feature_drop_indices = (input_table_select_feature_drop_indices - input_table_select_feature_stats.loc['mean']) / \
                                 input_table_select_feature_stats.loc['std']

    input_table_select_feature = pd.concat([input_table_select_feature_drop_indices,
                                              input_table_select_feature['Original indices']], axis=1)

    # Finally we tease out the target run from the input and the output
    mode_table_select_phase_target_run = mode_table_select_phase.loc[index_target_run]
    input_table_select_feature_target_run = input_table_select_feature.loc[index_target_run]
    # as well as the remaining rows in the input and the output
    if if_exclude_target_from_test>0:
        mode_table_select_phase_train_test = mode_table_select_phase.loc[~index_target_run]
        input_table_select_feature_train_test = input_table_select_feature.loc[~index_target_run]
    else:
        mode_table_select_phase_train_test = mode_table_select_phase
        input_table_select_feature_train_test = input_table_select_feature

    X = input_table_select_feature_train_test.values
    Y = mode_table_select_phase_train_test.values
    X_target_run = input_table_select_feature_target_run.values
    Y_target_run = mode_table_select_phase_target_run.values

    return X, Y, X_target_run, Y_target_run, input_table_select_feature_stats

def get_X_normalized_and_Y(XY_df, input_table_stats, select_target_run, references_lookup):

    # input table has first column as "Original indices"
    input_table, Y_table = XY_df

    target_run_Author, target_run_Year, if_exclude_target_from_test = select_target_run

    # select Ulmer et al. (2018)
    # Two steps. First, find the reference int he lookup table
    index_lookup_target_run_Author = references_lookup['references'].str.contains(target_run_Author)
    index_lookup_target_run_Year = references_lookup['references'].str.contains(target_run_Year)
    original_index_target_run = references_lookup.loc[
        index_lookup_target_run_Author & index_lookup_target_run_Year, 'index']
    # Second, select index in input_table_select_feature
    index_target_run = input_table['Original indices'].isin(original_index_target_run)

    # drop the "Original indices" column in input. because we do not need the indices in the NN input
    input_table_drop_indices = input_table.drop(columns='Original indices', axis=1)
    # normalization with mean and std

    # input_table_select_feature_stats = input_table_select_feature_drop_indices.describe()

    input_table_stats = input_table_stats.drop(columns='Original indices', axis=1)

    input_table_drop_indices = (input_table_drop_indices -
                                               input_table_stats.loc['mean']) / \
                                              input_table_stats.loc['std']

    input_table = pd.concat([input_table_drop_indices,
                                            input_table['Original indices']], axis=1)

    # Finally we tease out the target run from the input and the output
    Y_table_target_run = Y_table.loc[index_target_run]
    input_table_target_run = input_table.loc[index_target_run]
    # as well as the remaining rows in the input and the output
    if if_exclude_target_from_test > 0:
        Y_table_train_test = Y_table.loc[~index_target_run]
        input_table_train_test = input_table.loc[~index_target_run]
    else:
        Y_table_train_test = Y_table
        input_table_train_test = input_table

    X = input_table_train_test.values
    Y = Y_table_train_test.values
    X_target_run = input_table_target_run.values
    Y_target_run = Y_table_target_run.values

    return X, Y, X_target_run, Y_target_run, input_table_stats

def format_lepr_data_phase_mode(read_lepr_data, select_feature, y_range=[]):
    references_lookup, input_table, mode_table, phase_composition_table, norm_data = read_lepr_data
    select_mode_phases, select_input_features = select_feature

    # calculate sum of phase mode
    mode_sum = mode_table.sum(axis=1)
    # valid mode sum row indices
    index_select = mode_sum > 0

    # filter input and mode
    mode_table_select_phase = mode_table.loc[index_select, select_mode_phases]
    input_table_select_feature = input_table.loc[index_select, select_input_features]

    # get rid of samples with nan input values, usually nan occurs in the column of bulk 'H2O'
    row_has_nan = input_table_select_feature.isnull().any(axis=1)
    mode_table_select_phase = mode_table_select_phase[~row_has_nan]
    input_table_select_feature = input_table_select_feature[~row_has_nan]

    # specify the range of y values
    if len(y_range) == 2:
        row_y_constrained = mode_table_select_phase.iloc[:, 0].between(y_range[0], y_range[1], inclusive=True)
        mode_table_select_phase = mode_table_select_phase[row_y_constrained]
        input_table_select_feature = input_table_select_feature[row_y_constrained]

    return input_table_select_feature, mode_table_select_phase

def format_lepr_data_phase_composition(read_lepr_data, select_feature):

    references_lookup, input_table, mode_table, phase_composition_table, norm_data = read_lepr_data

    select_phases, select_input_features = select_feature

    # filter input and mode
    phase_compositions_columns_all = [[col for col in phase_composition_table.columns if phase in col] for phase in select_phases]
    phase_compositions_columns_all = [j for sub in phase_compositions_columns_all for j in sub]

    phase_compositions_columns_select = [[col for feature in select_input_features if feature in col]
                                         for col in phase_compositions_columns_all]

    phase_compositions_columns_select = [phase_compositions_columns_select[ii][0] for ii in
                                         range(len(phase_compositions_columns_select)) if
                                         phase_compositions_columns_select[ii]]

    phase_composition_table_select = phase_composition_table[phase_compositions_columns_select].copy()
    input_table_select_feature = input_table[select_input_features]

    # normalize anhydrous components to 100
    anhydrous_components_group_by_phase = [[col for col in phase_compositions_columns_select if (phase in col) and not ('H2O' in col)] for phase in select_phases]

    for anhydrous_components_phase in anhydrous_components_group_by_phase:
        anhydrous_components_phase_raw = phase_composition_table_select[anhydrous_components_phase] # returns a reference to the relevant columns in "phase_composition_table_select"
        anhydrous_components_phase_norm = anhydrous_components_phase_raw.div(anhydrous_components_phase_raw.sum(axis=1), axis=0)
        phase_composition_table_select[anhydrous_components_phase] = anhydrous_components_phase_norm.mul(100)
        # this operation occurs on the original dataframe "phase_composition_table_select"

    # get rid of samples with nan input values, usually nan occurs in the column of bulk 'H2O'
    row_has_nan = input_table_select_feature.isnull().any(axis=1)
    phase_composition_table_select = phase_composition_table_select[~row_has_nan]
    input_table_select_feature = input_table_select_feature[~row_has_nan]

    # get reference error for phase compositions
    phase_compositions_err = [[feature for feature in select_input_features if feature in col] for col in
              phase_compositions_columns_all]
    phase_compositions_columns_select_err = [phase_compositions_err[ii][0] for ii in
                                         range(len(phase_compositions_err)) if
                                         phase_compositions_err[ii]]

    phase_compositions_columns_select_err = norm_data.loc['err', phase_compositions_columns_select_err]

    return phase_compositions_columns_select, (input_table_select_feature, phase_composition_table_select), phase_compositions_columns_select_err


def get_XY_data_phase_mode(read_lepr_data, select_feature, select_target_run, y_range=[]):

    references_lookup, input_table, mode_table, phase_composition_table, norm_data = read_lepr_data

    select_phases, select_input_features = select_feature

    XY_df = format_lepr_data_phase_mode(read_lepr_data, select_feature, y_range=[])

    return get_X_normalized_and_Y(XY_df, norm_data[select_input_features], select_target_run, references_lookup)

def get_XY_data_phase_composition(read_lepr_data, select_feature, select_target_run, fill_fO2 = -1):

    references_lookup, input_table, mode_table, phase_composition_table, norm_data = read_lepr_data

    if fill_fO2:
        input_table['log fO2'] = input_table['log fO2'].replace(np.nan, 0)

    select_phases, select_input_features = select_feature

    phase_compositions_columns_select, XY_df, phase_compositions_columns_select_err = format_lepr_data_phase_composition(read_lepr_data, select_feature)

    X, Y, X_target_run, Y_target_run, input_stats = get_X_normalized_and_Y(XY_df, norm_data[select_input_features], select_target_run, references_lookup)

    return phase_compositions_columns_select, X, Y, X_target_run, Y_target_run, input_stats, phase_compositions_columns_select_err.values