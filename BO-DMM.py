import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.special import gamma
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from scipy.stats import pearsonr
import numpy as np
import joblib
import copy
import copy
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
RANDOM_SEED = 22
np.random.seed(RANDOM_SEED)
def load_data_excel(file_path):
    data = pd.read_excel(file_path)
    return data

def load_data(file_path, augment=False, augment_factor=1, angle_epsilon=10):
    data = pd.read_excel(file_path)
    # Remove rows with any NaN values
    data = data.dropna()
    # Exclude the last column for features and get target variable
    X = data.iloc[:, :-2].values.tolist()  # convert to nested list
    y = data['N'].values.tolist()
    return X, y, None, None


def reduce_dimensions(X, method='pca', n_components=5):
    if method == 'pca':
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X)
    elif method == 'elastic_net':
        elastic_net = ElasticNetCV(cv=5, random_state=42)
        elastic_net.fit(X, np.mean(X, axis=1))
        importance = np.abs(elastic_net.coef_)
        top_indices = np.argsort(importance)[-n_components:]
        return X[:, top_indices]
    else:
        raise ValueError("Unsupported dimensionality reduction method. Choose 'pca' or 'elastic_net'.")
def first_derivative(X):
    return np.gradient(X, axis=1)
def fractional_derivative(X, order=0.5, axis=0):
    """近似计算数组的分数阶导数。"""
    dX = np.zeros_like(X)
    points = 5
    for i in range(X.shape[axis]):
        idx_min = max(0, i - points)
        idx_max = min(X.shape[axis], i + points + 1)
        local_x = np.arange(idx_min, idx_max) - i
        local_y = X[idx_min:idx_max]
        weights = np.zeros_like(local_x, dtype=float)
        valid_mask = local_x != 0
        weights[valid_mask] = local_x[valid_mask] ** order
        weights /= gamma(order + 1)
        dX[i] = np.dot(weights, local_y)
    return dX



def preprocess_spectral_data(X, method='sg'):
    if method == 'sg':
        X_filtered = savgol_filter(X, window_length=11, polyorder=2, deriv=0, axis=0)
    elif method == 'cr':
        X_filtered = savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=0)
        X_filtered = X_filtered / np.max(X_filtered, axis=0)
    elif method == 'fd-r':
        X_filtered = np.gradient(X, axis=1)
    elif method == 'fod':
        X_filtered = fractional_derivative(X, order=0.5, axis=0)
    elif method == 'log':
        X_filtered = np.log1p(np.abs(X))
    elif method == 'reciprocal':
        X_filtered = 1.0 / (np.abs(X) + 1e-6)
    elif method == 'snv':
        X_mean = X.mean(axis=1)
        X_std = X.std(axis=1)
        X_filtered = (X - X_mean[:, np.newaxis]) / X_std[:, np.newaxis]
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X_filtered)
    return X_standardized

def select_high_corr_bands(X, y, num_features):
    correlations = np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
    top_indices = np.argsort(np.abs(correlations))[-num_features:][::-1]
    return top_indices


def select_spectral_bands(X, y, num_bands):
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X, y)
    importances = rf_regressor.feature_importances_
    top_indices = np.argsort(importances)[-num_bands:][::-1]
    return top_indices
#
# def calculate_influence_factors(diff_matrix):
#
#     U, S, VT = np.linalg.svd(diff_matrix, full_matrices=False)
#     influence_factors = VT[0]
#     return influence_factors

def calculate_influence_factors(diff_matrix):
    pca = PCA()
    pca.fit(diff_matrix)
    influence_factors = pca.components_[0]
    return influence_factors

def correction(spectra, influence_factors, alpha=10):
    corrected_spectra = spectra - alpha * np.outer(spectra, influence_factors)
    return corrected_spectra

def bayesian_correction(prior_mean, prior_variance, observed_spectra, observed_variance):
    posterior_mean = (prior_mean / prior_variance + observed_spectra / observed_variance) / (1 / prior_variance + 1 / observed_variance)
    posterior_variance = 1 / (1 / prior_variance + 1 / observed_variance)
    return posterior_mean, posterior_variance

def visualize_spectra(wavelength, dry_spectra, wet_spectra, corrected_spectra, sample_index, moisture_level):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, dry_spectra, label='Dry Spectrum', color='blue')
    plt.plot(wavelength, wet_spectra, label=f'{moisture_level}% Moisture', color='red')
    plt.plot(wavelength, corrected_spectra, label='Corrected Spectrum', color='green')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.title(f'Sample {sample_index + 1} Spectra at Different Moisture Levels')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_alpha(band_values):
    plsr_model = joblib.load('plsr_model.pkl')
    scaler = joblib.load('scaler.pkl')
    band_values = np.array(band_values).reshape(1, -1)
    band_values_scaled = scaler.transform(band_values)
    predicted_alpha = plsr_model.predict(band_values_scaled)
    return predicted_alpha[0]

def spa(X, y, num_features):
    correlations = np.abs(np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])]))
    initial_index = np.argmax(correlations)
    selected_indices = [initial_index]
    for _ in range(1, num_features):
        max_angle = -np.inf
        next_index = None
        for i in range(X.shape[1]):
            if i not in selected_indices:
                angles = [np.abs(np.dot(X[:, i], X[:, j]) / (np.linalg.norm(X[:, i]) * np.linalg.norm(X[:, j]))) for j
                          in selected_indices]
                min_angle = min(angles)
                if min_angle > max_angle:
                    max_angle = min_angle
                    next_index = i
        if next_index is not None:
            selected_indices.append(next_index)
    return selected_indices


def feature_selection(X, y, method, num_features):
    if method == 'pearson':
        indices = select_high_corr_bands(X, y, num_features)
    elif method == 'random_forest':
        indices = select_spectral_bands(X, y, num_features)
    elif method == 'spa':
        indices = spa(X, y, num_features)
    return indices


def apply_band_selection(X, indices):
    return X[:, indices]



def create_difference_matrix(spectra_0, spectra_wet_list):
    diff_matrix = []
    for spectra_wet in spectra_wet_list:
        for i in range(min(len(spectra_0), len(spectra_wet))):
            diff_matrix.append(spectra_wet[i] - spectra_0[i])
    return np.array(diff_matrix)

def calculate_rmse_for_two_spectra(dry_spectrum, corrected_spectrum):
    rmse = np.sqrt(np.mean((dry_spectrum - corrected_spectrum) ** 2))
    return rmse

def calculate_mscc_for_two_spectra(dry_spectrum, wet_spectrum):
    correlation_matrix = np.corrcoef(dry_spectrum, wet_spectrum)
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient


class CustomKFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.test_indices = np.arange(100)

    def split(self, X, y=None, groups=None):
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(self.test_indices)
        fold_sizes = np.full(self.n_splits, len(self.test_indices) // self.n_splits, dtype=int)
        fold_sizes[:len(self.test_indices) % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            test_index = self.test_indices[current:current + fold_size]
            train_index = np.setdiff1d(np.arange(len(X)), test_index)
            yield train_index, test_index
            current += fold_size


def calculate_percentage_rmse(y_true, y_pred):
    # 确保y_true中没有零值，因为我们需要除以这些值
    if np.any(y_true == 0):
        raise ValueError("y_true contains zero values, which will lead to division by zero in RMSE calculation.")
    relative_errors = (y_true - y_pred) / y_true
    squared_errors = relative_errors ** 2
    mean_squared_error = np.mean(squared_errors)
    percentage_rmse = np.sqrt(mean_squared_error) * 100
    return percentage_rmse
def compute_nrmse_mean(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_y = np.mean(y_true)
    nrmse = rmse / mean_y
    return nrmse


import numpy as np
import copy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, ttest_rel


def compute_nrmse_mean(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_y = np.mean(y_true)
    nrmse = rmse / mean_y
    return nrmse

from sklearn.model_selection import ShuffleSplit
def regression_model(X_train, y_train, spectra_6_selected, spectra_12_selected, spectra_18_selected,
                     spectra_24_selected,spectra_6_selected_r,spectra_12_selected_r,spectra_18_selected_r,spectra_24_selected_r,
                     model_type='svm', n_splits=5, data_agument=True):
    # 模型选择部分保持不变
    if model_type == 'pls':
        from sklearn.cross_decomposition import PLSRegression
        model = PLSRegression(n_components=4, scale=True)
    elif model_type == 'svm':
        from sklearn.svm import SVR
        model = SVR(kernel='rbf', gamma='scale')
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, random_state=13)
    else:
        raise ValueError("Unsupported model type. Choose 'pls', 'svm', or 'random_forest'.")

    # 数据转换
    X_train_selected = np.array(X_train)
    y_train = np.array(y_train)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=27)

    # 初始化结果字典，添加 MAE
    cv_results = {
        "r2_scores_0": [], "rmse_scores_0": [], "pearson_correlations_0": [], "mae_scores_0": [],
        "r2_scores_6": [], "rmse_scores_6": [], "pearson_correlations_6": [], "mae_scores_6": [],
        "r2_scores_12": [], "rmse_scores_12": [], "pearson_correlations_12": [], "mae_scores_12": [],
        "r2_scores_18": [], "rmse_scores_18": [], "pearson_correlations_18": [], "mae_scores_18": [],
        "r2_scores_24": [], "rmse_scores_24": [], "pearson_correlations_24": [], "mae_scores_24": [],
        "mae_dry": [], "mae_moist": []  # 新增 Dry 和 Moist 的 MAE
    }
    cvo_results = {
        "o_r2_scores_6": [], "o_rmse_scores_6": [], "o_pearson_correlations_6": [], "o_mae_scores_6": [],
        "o_r2_scores_12": [], "o_rmse_scores_12": [], "o_pearson_correlations_12": [], "o_mae_scores_12": [],
        "o_r2_scores_18": [], "o_rmse_scores_18": [], "o_pearson_correlations_18": [], "o_mae_scores_18": [],
        "o_r2_scores_24": [], "o_rmse_scores_24": [], "o_pearson_correlations_24": [], "o_mae_scores_24": []
    }
    corrected_spectra = {
        'corrected_0': [],
        'corrected_6': [],
        'corrected_12': [],
        'corrected_18': [],
        'corrected_24': []
    }
    y_true = []
    y_predict_dry = []
    y_predict_moist = []
    y_predict_correct = []
    X_test = copy.deepcopy(X_train_selected[0:100])

    error_orig_all = []
    error_correct_all = []
    splitter = ShuffleSplit(n_splits=5, test_size=50, train_size=50, random_state=13)

    if data_agument:
        for split_num, (train_index, test_index) in enumerate(splitter.split(X_test), 1):
            y_train_f = copy.deepcopy(y_train[100:])
            X_train_fold = copy.deepcopy(X_train_selected[100:])
            X_test_fold = copy.deepcopy(X_train_selected[test_index])
            X_test_6_fold = copy.deepcopy(spectra_6_selected_r[test_index])
            X_test_12_fold = copy.deepcopy(spectra_12_selected_r[test_index])
            X_test_18_fold = copy.deepcopy(spectra_18_selected_r[test_index])
            X_test_24_fold = copy.deepcopy(spectra_24_selected_r[test_index])
            spectra_0_car = copy.deepcopy(X_train_selected[train_index])
            spectra_6_train = copy.deepcopy(spectra_6_selected[train_index])
            spectra_12_train = copy.deepcopy(spectra_12_selected[train_index])
            spectra_18_train = copy.deepcopy(spectra_18_selected[train_index])
            spectra_24_train = copy.deepcopy(spectra_24_selected[train_index])
            diff_matrix = create_difference_matrix(spectra_0_car, [spectra_6_train, spectra_12_train, spectra_18_train,
                                                                   spectra_24_train])
            influence_factors = calculate_influence_factors(diff_matrix)
            prior_mean = np.mean(X_train_fold, axis=0)
            prior_variance = np.var(X_train_fold, axis=0)
            corrected_0_test = []
            corrected_6_test = []
            corrected_12_test = []
            corrected_18_test = []
            corrected_24_test = []
            sam_results = {6: [], 12: [], 18: [], 24: []}
            for i in range(len(X_test_6_fold)):
                alpha_6 = predict_alpha(X_test_6_fold[i:i + 1])
                alpha_12 = predict_alpha(X_test_12_fold[i:i + 1])
                alpha_18 = predict_alpha(X_test_18_fold[i:i + 1])
                alpha_24 = predict_alpha(X_test_24_fold[i:i + 1])
                corrected_6 = correction(X_test_6_fold[i:i + 1], influence_factors, alpha_6)
                corrected_12 = correction(X_test_12_fold[i:i + 1], influence_factors, alpha_12)
                corrected_18 = correction(X_test_18_fold[i:i + 1], influence_factors, alpha_18)
                corrected_24 = correction(X_test_24_fold[i:i + 1], influence_factors, alpha_24)
                posterior_mean_6, _ = bayesian_correction(prior_mean, prior_variance, corrected_6[0],
                                                          np.var(X_test_6_fold, axis=0))
                posterior_mean_12, _ = bayesian_correction(prior_mean, prior_variance, corrected_12[0],
                                                           np.var(X_test_12_fold, axis=0))
                posterior_mean_18, _ = bayesian_correction(prior_mean, prior_variance, corrected_18[0],
                                                           np.var(X_test_18_fold, axis=0))
                posterior_mean_24, _ = bayesian_correction(prior_mean, prior_variance, corrected_24[0],
                                                           np.var(X_test_24_fold, axis=0))
                corrected_0_test.append(X_test_fold[i:i + 1][0])
                corrected_6_test.append(posterior_mean_6)
                corrected_12_test.append(posterior_mean_12)
                corrected_18_test.append(posterior_mean_18)
                corrected_24_test.append(posterior_mean_24)
                sam_6_original = calculate_mscc_for_two_spectra(X_test_fold[i:i + 1][0], X_test_6_fold[i:i + 1][0])
                sam_6_corrected_epo = calculate_mscc_for_two_spectra(X_test_fold[i:i + 1][0], corrected_6[0])
                sam_results[6].append((sam_6_original, sam_6_corrected_epo))
                sam_12_original = calculate_mscc_for_two_spectra(X_test_fold[i:i + 1][0], X_test_12_fold[i:i + 1][0])
                sam_12_corrected_epo = calculate_mscc_for_two_spectra(X_test_fold[i:i + 1][0], corrected_12[0])
                sam_results[12].append((sam_12_original, sam_12_corrected_epo))
                sam_18_original = calculate_mscc_for_two_spectra(X_test_fold[i:i + 1][0], X_test_18_fold[i:i + 1][0])
                sam_18_corrected_epo = calculate_mscc_for_two_spectra(X_test_fold[i:i + 1][0], corrected_18[0])
                sam_results[18].append((sam_18_original, sam_18_corrected_epo))
                sam_24_original = calculate_mscc_for_two_spectra(X_test_fold[i:i + 1][0], X_test_24_fold[i:i + 1][0])
                sam_24_corrected_epo = calculate_mscc_for_two_spectra(X_test_fold[i:i + 1][0], corrected_24[0])
                sam_results[24].append((sam_24_original, sam_24_corrected_epo))
            corrected_0_test = np.array(corrected_0_test)
            corrected_6_test = np.array(corrected_6_test)
            corrected_12_test = np.array(corrected_12_test)
            corrected_18_test = np.array(corrected_18_test)
            corrected_24_test = np.array(corrected_24_test)
            preprocess_method = 'snv'
            X_test_fold_preprocessed = preprocess_spectral_data(corrected_0_test, method=preprocess_method)
            X_train_preprocessed = preprocess_spectral_data(X_train_fold, method=preprocess_method)
            spectra_6_preprocessed = preprocess_spectral_data(X_test_6_fold, method=preprocess_method)
            spectra_12_preprocessed = preprocess_spectral_data(X_test_12_fold, method=preprocess_method)
            spectra_18_preprocessed = preprocess_spectral_data(X_test_18_fold, method=preprocess_method)
            spectra_24_preprocessed = preprocess_spectral_data(X_test_24_fold, method=preprocess_method)
            spectra_6_preprocessed_correct = preprocess_spectral_data(corrected_6_test, method=preprocess_method)
            spectra_12_preprocessed_correct = preprocess_spectral_data(corrected_12_test, method=preprocess_method)
            spectra_18_preprocessed_correct = preprocess_spectral_data(corrected_18_test, method=preprocess_method)
            spectra_24_preprocessed_correct = preprocess_spectral_data(corrected_24_test, method=preprocess_method)
            X_train_fold = X_train_preprocessed
            y_train_fold = y_train_f
            y_test_fold = y_train[test_index]
            # print(X_train_fold.shape)
            model.fit(X_train_fold, y_train_fold)
            predictions_fold_0 = model.predict(X_test_fold_preprocessed)
            o_predictions_fold_6 = model.predict(spectra_6_preprocessed)
            o_predictions_fold_12 = model.predict(spectra_12_preprocessed)
            o_predictions_fold_18 = model.predict(spectra_18_preprocessed)
            o_predictions_fold_24 = model.predict(spectra_24_preprocessed)
            predictions_fold_6 = model.predict(spectra_6_preprocessed_correct)
            predictions_fold_12 = model.predict(spectra_12_preprocessed_correct)
            predictions_fold_18 = model.predict(spectra_18_preprocessed_correct)
            predictions_fold_24 = model.predict(spectra_24_preprocessed_correct)
            r2_0 = r2_score(y_test_fold, predictions_fold_0)
            r2_6 = r2_score(y_test_fold, predictions_fold_6)
            r2_12 = r2_score(y_test_fold, predictions_fold_12)
            r2_18 = r2_score(y_test_fold, predictions_fold_18)
            r2_24 = r2_score(y_test_fold, predictions_fold_24)
            o_r2_6 = r2_score(y_test_fold, o_predictions_fold_6)
            o_r2_12 = r2_score(y_test_fold, o_predictions_fold_12)
            o_r2_18 = r2_score(y_test_fold, o_predictions_fold_18)
            o_r2_24 = r2_score(y_test_fold, o_predictions_fold_24)
            mse_0 = mean_squared_error(y_test_fold, predictions_fold_0)
            o_mse_6 = mean_squared_error(y_test_fold, o_predictions_fold_6)
            o_mse_12 = mean_squared_error(y_test_fold, o_predictions_fold_12)
            o_mse_18 = mean_squared_error(y_test_fold, o_predictions_fold_18)
            o_mse_24 = mean_squared_error(y_test_fold, o_predictions_fold_24)
            mse_6 = mean_squared_error(y_test_fold, predictions_fold_6)
            mse_12 = mean_squared_error(y_test_fold, predictions_fold_12)
            mse_18 = mean_squared_error(y_test_fold, predictions_fold_18)
            mse_24 = mean_squared_error(y_test_fold, predictions_fold_24)
            rmse_0 = np.sqrt(mse_0)
            nrmse_0_range = rmse_0 / (max(y_test_fold) - min(y_test_fold))
            nrmse_0_mean = rmse_0 / np.mean(y_test_fold)
            rmse_6 = np.sqrt(mse_6)
            nrmse_6_range = rmse_6 / (max(y_test_fold) - min(y_test_fold))
            nrmse_6_mean = rmse_6 / np.mean(y_test_fold)
            rmse_12 = np.sqrt(mse_12)
            nrmse_12_range = rmse_12 / (max(y_test_fold) - min(y_test_fold))
            nrmse_12_mean = rmse_12 / np.mean(y_test_fold)
            rmse_18 = np.sqrt(mse_18)
            nrmse_18_range = rmse_18 / (max(y_test_fold) - min(y_test_fold))
            nrmse_18_mean = rmse_18 / np.mean(y_test_fold)
            rmse_24 = np.sqrt(mse_24)
            nrmse_24_range = rmse_24 / (max(y_test_fold) - min(y_test_fold))
            nrmse_24_mean = rmse_24 / np.mean(y_test_fold)
            o_rmse_6 = np.sqrt(o_mse_6)
            o_nrmse_6_range = o_rmse_6 / (max(y_test_fold) - min(y_test_fold))
            o_nrmse_6_mean = o_rmse_6 / np.mean(y_test_fold)
            o_rmse_12 = np.sqrt(o_mse_12)
            o_nrmse_12_range = o_rmse_12 / (max(y_test_fold) - min(y_test_fold))
            o_nrmse_12_mean = o_rmse_12 / np.mean(y_test_fold)
            o_rmse_18 = np.sqrt(o_mse_18)
            o_nrmse_18_range = o_rmse_18 / (max(y_test_fold) - min(y_test_fold))
            o_nrmse_18_mean = o_rmse_18 / np.mean(y_test_fold)
            o_rmse_24 = np.sqrt(o_mse_24)
            o_nrmse_24_range = o_rmse_24 / (max(y_test_fold) - min(y_test_fold))
            o_nrmse_24_mean = o_rmse_24 / np.mean(y_test_fold)
            rmse_0 = nrmse_0_mean
            rmse_6 = nrmse_6_mean
            rmse_12 = nrmse_12_mean
            rmse_18 = nrmse_18_mean
            rmse_24 = nrmse_24_mean
            o_rmse_6 = o_nrmse_6_mean
            o_rmse_12 = o_nrmse_12_mean
            o_rmse_18 = o_nrmse_18_mean
            o_rmse_24 = o_nrmse_24_mean
            pearson_corr_0, _ = pearsonr(y_test_fold, predictions_fold_0)
            pearson_corr_6, _ = pearsonr(y_test_fold, predictions_fold_6)
            pearson_corr_12, _ = pearsonr(y_test_fold, predictions_fold_12)
            pearson_corr_18, _ = pearsonr(y_test_fold, predictions_fold_18)
            pearson_corr_24, _ = pearsonr(y_test_fold, predictions_fold_24)
            o_pearson_corr_6, _ = pearsonr(y_test_fold, o_predictions_fold_6)
            o_pearson_corr_12, _ = pearsonr(y_test_fold, o_predictions_fold_12)
            o_pearson_corr_18, _ = pearsonr(y_test_fold, o_predictions_fold_18)
            o_pearson_corr_24, _ = pearsonr(y_test_fold, o_predictions_fold_24)


            cv_results["r2_scores_0"].append(r2_0)
            cv_results["rmse_scores_0"].append(rmse_0)
            cv_results["pearson_correlations_0"].append(pearson_corr_0)
            cv_results["r2_scores_6"].append(r2_6)
            cv_results["rmse_scores_6"].append(rmse_6)
            cv_results["pearson_correlations_6"].append(pearson_corr_6)
            cv_results["r2_scores_12"].append(r2_12)
            cv_results["rmse_scores_12"].append(rmse_12)
            cv_results["pearson_correlations_12"].append(pearson_corr_12)
            cv_results["r2_scores_18"].append(r2_18)
            cv_results["rmse_scores_18"].append(rmse_18)
            cv_results["pearson_correlations_18"].append(pearson_corr_18)
            cv_results["r2_scores_24"].append(r2_24)
            cv_results["rmse_scores_24"].append(rmse_24)
            cv_results["pearson_correlations_24"].append(pearson_corr_24)
            cvo_results["o_r2_scores_6"].append(o_r2_6)
            cvo_results["o_rmse_scores_6"].append(o_rmse_6)
            cvo_results["o_pearson_correlations_6"].append(o_pearson_corr_6)
            cvo_results["o_r2_scores_12"].append(o_r2_12)
            cvo_results["o_rmse_scores_12"].append(o_rmse_12)
            cvo_results["o_pearson_correlations_12"].append(o_pearson_corr_12)
            cvo_results["o_r2_scores_18"].append(o_r2_18)
            cvo_results["o_rmse_scores_18"].append(o_rmse_18)
            cvo_results["o_pearson_correlations_18"].append(o_pearson_corr_18)
            cvo_results["o_r2_scores_24"].append(o_r2_24)
            cvo_results["o_rmse_scores_24"].append(o_rmse_24)
            cvo_results["o_pearson_correlations_24"].append(o_pearson_corr_24)

            corrected_spectra['corrected_0'].extend(corrected_0_test.tolist())
            corrected_spectra['corrected_6'].extend(corrected_6_test.tolist())
            corrected_spectra['corrected_12'].extend(corrected_12_test.tolist())
            corrected_spectra['corrected_18'].extend(corrected_18_test.tolist())
            corrected_spectra['corrected_24'].extend(corrected_24_test.tolist())


            y_test_fold = list(y_test_fold)
            predictions_fold_0 = list(predictions_fold_0)
            o_predictions_fold_12 = list(o_predictions_fold_12)
            predictions_fold_12 = list(predictions_fold_12)
            y_true = y_true + y_test_fold
            y_predict_dry = y_predict_dry + predictions_fold_0
            y_predict_moist = y_predict_moist + o_predictions_fold_12
            y_predict_correct = y_predict_correct + predictions_fold_12


            error_orig = np.abs(np.array(predictions_fold_0) - np.array(y_test_fold))
            error_correct = np.abs(np.array(predictions_fold_18) - np.array(y_test_fold))
            error_orig_all.extend(error_orig)
            error_correct_all.extend(error_correct)


    model.fit(X_train_selected, y_train)
    predictions_test = model.predict(X_train_selected)
    print(cv_results["r2_scores_0"])


    average_r2 = np.mean(cv_results["r2_scores_0"])
    average_rmse = np.mean(cv_results["rmse_scores_0"])
    average_pearson = np.mean(cv_results["pearson_correlations_0"])
    average_mae = np.mean(cv_results["mae_scores_0"])



    mae_dry = mean_absolute_error(y_true, y_predict_dry)
    mae_moist = mean_absolute_error(y_true, y_predict_moist)
    mae_correct = mean_absolute_error(y_true, y_predict_correct)


    if len(error_orig_all) > 0 and len(error_correct_all) > 0:
        error_orig_all = np.array(error_orig_all)
        error_correct_all = np.array(error_correct_all)

        t_statistic, p_value = ttest_rel(error_orig_all, error_correct_all)

    else:
        p_value = None

    return (
        np.mean(cv_results["r2_scores_0"]),
        np.mean(cv_results["rmse_scores_0"]),
        np.mean(cv_results["pearson_correlations_0"]),
        np.mean(cv_results["mae_scores_0"]),
        predictions_test,
        np.mean(cv_results["r2_scores_6"]),
        np.mean(cv_results["rmse_scores_6"]),
        np.mean(cv_results["pearson_correlations_6"]),
        np.mean(cv_results["mae_scores_6"]),
        np.mean(cv_results["r2_scores_12"]),
        np.mean(cv_results["rmse_scores_12"]),
        np.mean(cv_results["pearson_correlations_12"]),
        np.mean(cv_results["mae_scores_12"]),
        np.mean(cv_results["r2_scores_18"]),
        np.mean(cv_results["rmse_scores_18"]),
        np.mean(cv_results["pearson_correlations_18"]),
        np.mean(cv_results["mae_scores_18"]),
        np.mean(cv_results["r2_scores_24"]),
        np.mean(cv_results["rmse_scores_24"]),
        np.mean(cv_results["pearson_correlations_24"]),
        np.mean(cv_results["mae_scores_24"]),
        np.mean(cvo_results["o_r2_scores_6"]),
        np.mean(cvo_results["o_rmse_scores_6"]),
        np.mean(cvo_results["o_pearson_correlations_6"]),
        np.mean(cvo_results["o_mae_scores_6"]),
        np.mean(cvo_results["o_r2_scores_12"]),
        np.mean(cvo_results["o_rmse_scores_12"]),
        np.mean(cvo_results["o_pearson_correlations_12"]),
        np.mean(cvo_results["o_mae_scores_12"]),
        np.mean(cvo_results["o_r2_scores_18"]),
        np.mean(cvo_results["o_rmse_scores_18"]),
        np.mean(cvo_results["o_pearson_correlations_18"]),
        np.mean(cvo_results["o_mae_scores_18"]),
        np.mean(cvo_results["o_r2_scores_24"]),
        np.mean(cvo_results["o_rmse_scores_24"]),
        np.mean(cvo_results["o_pearson_correlations_24"]),
        np.mean(cvo_results["o_mae_scores_24"]),
        y_true,
        y_predict_dry,
        y_predict_moist,
        y_predict_correct,
        corrected_spectra,
        p_value
    )


data_6 = load_data_excel(r"6%.xlsx")
data_12 = load_data_excel(r"12%.xlsx")
data_18 = load_data_excel(r"18%.xlsx")
data_24 = load_data_excel(r"24%.xlsx")
data_6r = load_data_excel(r"1.xlsx")
data_12r = load_data_excel(r"2.xlsx")
data_18r = load_data_excel(r"3.xlsx")
data_24r = load_data_excel(r"4.xlsx")


spectra_6 = data_6.iloc[:, 1:-1]
spectra_12 = data_12.iloc[:, 1:-1]
spectra_18 = data_18.iloc[:, 1:-1]
spectra_24 = data_24.iloc[:, 1:-1]
spectra_6 = spectra_6.iloc[:, :].values.tolist()
spectra_12 = spectra_12.iloc[:, :].values.tolist()
spectra_18 = spectra_18.iloc[:, :].values.tolist()
spectra_24 = spectra_24.iloc[:, :].values.tolist()
spectra_6 = np.array(spectra_6)
spectra_12 = np.array(spectra_12)
spectra_18 = np.array(spectra_18)
spectra_24 = np.array(spectra_24)
spectra_6_r = data_6r.iloc[:, 1:-1]
spectra_12_r = data_12r.iloc[:, 1:-1]
spectra_18_r = data_18r.iloc[:, 1:-1]
spectra_24_r = data_24r.iloc[:, 1:-1]
spectra_6_r = spectra_6_r.iloc[:, :].values.tolist()
spectra_12_r = spectra_12_r.iloc[:, :].values.tolist()
spectra_18_r = spectra_18_r.iloc[:, :].values.tolist()
spectra_24_r = spectra_24_r.iloc[:, :].values.tolist()
spectra_6_r = np.array(spectra_6_r)
spectra_12_r = np.array(spectra_12_r)
spectra_18_r = np.array(spectra_18_r)
spectra_24_r = np.array(spectra_24_r)
train_path = r"0%.xlsx"
X_train, y_train, X_augmented, y_augmented = load_data(train_path, augment=False)




model_type = 'random_forest'
(
    mean_r2_cv_0,
    mean_rmse_cv_0,
    mean_pearson_cv_0,
    mean_mae_cv_0,
    predictions_test,
    mean_r2_cv_6,
    mean_rmse_cv_6,
    mean_pearson_cv_6,
    mean_mae_cv_6,
    mean_r2_cv_12,
    mean_rmse_cv_12,
    mean_pearson_cv_12,
    mean_mae_cv_12,
    mean_r2_cv_18,
    mean_rmse_cv_18,
    mean_pearson_cv_18,
    mean_mae_cv_18,
    mean_r2_cv_24,
    mean_rmse_cv_24,
    mean_pearson_cv_24,
    mean_mae_cv_24,
    o_mean_r2_cv_6,
    o_mean_rmse_cv_6,
    o_mean_pearson_cv_6,
    o_mean_mae_cv_6,
    o_mean_r2_cv_12,
    o_mean_rmse_cv_12,
    o_mean_pearson_cv_12,
    o_mean_mae_cv_12,
    o_mean_r2_cv_18,
    o_mean_rmse_cv_18,
    o_mean_pearson_cv_18,
    o_mean_mae_cv_18,
    o_mean_r2_cv_24,
    o_mean_rmse_cv_24,
    o_mean_pearson_cv_24,
    o_mean_mae_cv_24,
    y_true,
    y_predict_dry,
    y_predict_moist,
    y_predict_correct,
    corrected_spectra,
    p_value  # 新增 p_value
) = regression_model(
    X_train, y_train,
    spectra_6, spectra_12, spectra_18, spectra_24,spectra_6_r,spectra_12_r,spectra_18_r,spectra_24_r,
    model_type=model_type,
    n_splits=5,
    data_agument=True
)

print("pair t p_value",p_value)
original_spectra = {
    '0': X_train,
    '6': spectra_6,
    '12': spectra_12,
    '18': spectra_18,
    '24': spectra_24
}

r2_cv = [mean_r2_cv_6, mean_r2_cv_24, mean_r2_cv_18, mean_r2_cv_12]
rmse_cv = [mean_rmse_cv_6, mean_rmse_cv_24, mean_rmse_cv_18, mean_rmse_cv_12]
pearson_cv = [mean_pearson_cv_6, mean_pearson_cv_24, mean_pearson_cv_18, mean_pearson_cv_12]
mae_cv = [mean_mae_cv_6, mean_mae_cv_24, mean_mae_cv_18, mean_mae_cv_12]

o_r2_cv = [o_mean_r2_cv_6, o_mean_r2_cv_24, o_mean_r2_cv_18, o_mean_r2_cv_12]
o_rmse_cv = [o_mean_rmse_cv_6, o_mean_rmse_cv_24, o_mean_rmse_cv_18, o_mean_rmse_cv_12]
o_pearson_cv = [o_mean_pearson_cv_6, o_mean_pearson_cv_24, o_mean_pearson_cv_18, o_mean_pearson_cv_12]
o_mae_cv = [o_mean_mae_cv_6, o_mean_mae_cv_24, o_mean_mae_cv_18, o_mean_mae_cv_12]


average_r2 = np.mean(r2_cv)
average_rmse = np.mean(rmse_cv)
average_pearson = np.mean(pearson_cv)
average_mae = np.mean(mae_cv)

o_average_r2 = np.mean(o_r2_cv)
o_average_rmse = np.mean(o_rmse_cv)
o_average_pearson = np.mean(o_pearson_cv)
o_average_mae = np.mean(o_mae_cv)


mae_dry = mean_absolute_error(y_true, y_predict_dry)
mae_moist = mean_absolute_error(y_true, y_predict_moist)
mae_correct = mean_absolute_error(y_true, y_predict_correct)

r2_cv = [mean_r2_cv_6, mean_r2_cv_12,mean_r2_cv_18, mean_r2_cv_24]
rmse_cv = [mean_rmse_cv_6, mean_rmse_cv_12,mean_rmse_cv_18, mean_rmse_cv_24]
pearson_cv = [mean_pearson_cv_6, mean_pearson_cv_12,mean_pearson_cv_18, mean_pearson_cv_24]

o_r2_cv = [o_mean_r2_cv_6,o_mean_r2_cv_12, o_mean_r2_cv_18, o_mean_r2_cv_24]
o_rmse_cv = [o_mean_rmse_cv_6,o_mean_rmse_cv_12, o_mean_rmse_cv_18, o_mean_rmse_cv_24]
o_pearson_cv = [o_mean_pearson_cv_6,o_mean_pearson_cv_12, o_mean_pearson_cv_18, o_mean_pearson_cv_24]

average_r2 = sum(r2_cv) / len(r2_cv)
average_rmse = sum(rmse_cv) / len(rmse_cv)
average_pearson = sum(pearson_cv) / len(pearson_cv)

o_average_r2 = sum(o_r2_cv) / len(o_r2_cv)
o_average_rmse = sum(o_rmse_cv) / len(o_rmse_cv)
o_average_pearson = sum(o_pearson_cv) / len(o_pearson_cv)
def plot_results(y_test, predictions, r2, rmse, pearson,correct):
    plt.figure(figsize=(8, 6))


    plt.scatter(y_test, predictions, color='blue', alpha=0.7)


    min_val = min(min(y_test), min(predictions))
    max_val = max(max(y_test), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', lw=2)


    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "k--", lw=2)
    nc_variable = str(correct)
    # Text box with metrics
    textstr = '\n'.join((
        fr'$\mathrm{{{nc_variable}\text{{-}}svm}}$',
        r'$R^2=%.3f$' % (r2,),
        r'$\mathrm{nRMSE}=%.3f$' % (rmse,),
        r'$\mathrm{pearson}=%.3f\ $' % (pearson,)))


    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
             verticalalignment='top', bbox=props)
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Measured SOM ($\\times 1 \\mathrm{g\ kg^{-1}}$)')
    plt.ylabel('Predicted SOM ($\\times 1 \\mathrm{g\ kg^{-1}}$)')
    plt.grid(True)
    plt.show()
print('correct', average_r2, 'origin', o_average_r2)
print('correct', average_rmse, 'origin', o_average_rmse)
print('correct', average_pearson, 'origin', o_average_pearson)
plot_results(y_true, y_predict_dry, mean_r2_cv_0, mean_rmse_cv_0, mean_pearson_cv_0 ,'Dry')
plot_results(y_true, y_predict_moist, o_average_r2, o_average_rmse, o_average_pearson,  'NC')
plot_results(y_true, y_predict_correct, average_r2, average_rmse, average_pearson, 'BO-DMM-pca')
print('Dry MAE:', mae_dry)
print('Moist MAE:', mae_moist)
print('Corrected MAE:', mae_correct)

