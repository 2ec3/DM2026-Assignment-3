import math
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time

TRAIN_FILE_NUMBERS = 11020
TRAIN_USER_NUMBERS = 60

TEST_FILE_NUMBERS = 6849
TEST_USER_NUMBERS = 40

# -------------------------------------------------- file & display --------------------------------------------------

def continuous_Bayesian_classifier_to_file(file_name, classifier):
    file = open("baseline_methods\\" + file_name, "w")

    number_classes = len(classifier)
    number_features = len(classifier[0]) - 1
    number_seconds = len(classifier[0][0])

    file.write(str(number_classes) + "\n")
    file.write(str(number_features) + "\n")
    file.write(str(number_seconds) + "\n" + "\n")

    for c in range(number_classes):
        class_c = classifier[c]
        file.write(str(class_c[number_features]) + "\n")
        for f in range(number_features):
            feature_f = class_c[f]
            for s in range(number_seconds):
                separator = "|"
                if (s == number_seconds - 1):
                    separator = "\n"
                    if (f == number_features - 1):
                        separator += "\n"
                file.write(str(feature_f[s][0]) + "," + str(feature_f[s][1]) + separator)

    file.close()

def file_to_continuous_Bayesian_classifier(file_name):
    file = open("baseline_methods\\" + file_name, "r")

    number_classes = int(file.readline())
    number_features = int(file.readline())
    number_seconds = int(file.readline())
    
    file.readline()

    classifier = []

    for c in range(number_classes):
        n = int(file.readline())
        class_c = []
        for f in range(number_features):
            line = file.readline().split("|")
            feature_f = []
            for s in range(number_seconds):
                file_mle = line[s].split(",")
                mle = [float(file_mle[0]), float(file_mle[1])]
                feature_f.append(mle)
            class_c.append(feature_f)
        class_c.append(n)
        classifier.append(class_c)
        file.readline()

    return classifier

def file_csv_to_df(train, file_id, user_id = 0, g_to_ms2 = 9.81):
    if (train):
        assert 1 <= file_id and file_id <= TRAIN_FILE_NUMBERS
        assert 0 <= user_id and user_id <= TRAIN_USER_NUMBERS
        folder = "train"
    else:
        assert TRAIN_FILE_NUMBERS + 1 <= file_id and file_id <= TRAIN_FILE_NUMBERS + TEST_FILE_NUMBERS
        assert TRAIN_USER_NUMBERS <= user_id and user_id <= TRAIN_USER_NUMBERS + TEST_USER_NUMBERS
        folder = "test"
    
    if ((train and user_id != 0) or (not train and user_id != 60)):
        df = pd.read_csv(f'data/{folder}/User_{user_id:03d}/{file_id:05d}.csv')
        df['mean_x'] *= g_to_ms2
        df['mean_y'] *= g_to_ms2
        df['mean_z'] *= g_to_ms2
        df['std_x'] *= g_to_ms2
        df['std_y'] *= g_to_ms2
        df['std_z'] *= g_to_ms2
        return df
    
    pattern = f"data/{folder}/User_*/{file_id:05d}.csv"
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for file_id={file_id}")

    df = pd.read_csv(matches[0])
    df['mean_x'] *= g_to_ms2
    df['mean_y'] *= g_to_ms2
    df['mean_z'] *= g_to_ms2
    df['std_x'] *= g_to_ms2
    df['std_y'] *= g_to_ms2
    df['std_z'] *= g_to_ms2
    return df

def user_csv_to_dfs(train, user_id, g_to_ms2 = 9.81):
    if (train):
        assert 1 <= user_id and user_id <= TRAIN_USER_NUMBERS
        folder = "train"
    else:
        assert TRAIN_USER_NUMBERS + 1 <= user_id and user_id <= TRAIN_USER_NUMBERS + TEST_USER_NUMBERS
        folder = "test"

    pattern = f"data/{folder}/User_{user_id:03d}/*.csv"
    matches = glob.glob(pattern)

    if len(matches) == 0:
        raise FileNotFoundError(f"No file found for user_id={user_id}")

    dfs = []

    for csv in matches:
        df = pd.read_csv(csv)
        df['mean_x'] *= g_to_ms2
        df['mean_y'] *= g_to_ms2
        df['mean_z'] *= g_to_ms2
        df['std_x'] *= g_to_ms2
        df['std_y'] *= g_to_ms2
        df['std_z'] *= g_to_ms2
        
        dfs.append(df)
    
    return dfs

def add_xyz_speed_position(df):
    x_means = df['mean_x'].to_numpy()
    y_means = df['mean_y'].to_numpy()
    z_means = df['mean_z'].to_numpy()

    x_speeds = [x_means[0]]
    y_speeds = [y_means[0]]
    z_speeds = [z_means[0]]

    x_positions = [x_speeds[0]]
    y_positions = [y_speeds[0]]
    z_positions = [z_speeds[0]]

    for i in range(1, len(x_means)):
        x_speeds.append(x_speeds[i - 1] + x_means[i])
        y_speeds.append(y_speeds[i - 1] + y_means[i])
        z_speeds.append(z_speeds[i - 1] + z_means[i])

        x_positions.append(x_positions[i - 1] + x_speeds[i])
        y_positions.append(y_positions[i - 1] + y_speeds[i])
        z_positions.append(z_positions[i - 1] + z_speeds[i])
    
    df['speed_x'] = x_speeds
    df['speed_y'] = y_speeds
    df['speed_z'] = z_speeds

    df['position_x'] = x_positions
    df['position_y'] = y_positions
    df['position_z'] = z_positions

    return df

def data_frame_accelerations_to_figures(df):
    fig, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=True)

    seconds = df['index']

    axes_map = {
        0: ('x axis', 'mean_x', 'std_x'),
        1: ('y axis', 'mean_y', 'std_y'),
        2: ('z axis', 'mean_z', 'std_z'),
    }

    for i, (label, mean_col, std_col) in axes_map.items():
        mean = df[mean_col]
        std = df[std_col]

        ax = axes[i]

        ax.plot(seconds, mean, color='blue', label='mean', lw = 0.5)

        ax.fill_between(
            seconds,
            mean - std,
            mean + std,
            color='blue',
            alpha=0.2,
            label='± std'
        )

        ax.set_title(label)
        ax.legend()

    fig.supylabel("acceleration [m/s²]")
    fig.supxlabel("time [seconds]")

    plt.tight_layout()
    plt.show()

def data_frame_speeds_to_figures(df):
    fig, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=True)

    seconds = df['index']

    axes_map = {
        0: ('x axis', 'speed_x'),
        1: ('y axis', 'speed_y'),
        2: ('z axis', 'speed_z'),
    }

    for i, (label, speed_col) in axes_map.items():
        speed = df[speed_col]

        ax = axes[i]

        ax.plot(seconds, speed, color='blue', lw = 0.5)

        ax.set_title(label)

    fig.supylabel("speed [m/s]")
    fig.supxlabel("time [seconds]")

    plt.tight_layout()
    plt.show()

def data_frame_positions_to_figures(df):
    fig, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=True)

    seconds = df['index']

    axes_map = {
        0: ('x axis', 'position_x'),
        1: ('y axis', 'position_y'),
        2: ('z axis', 'position_z'),
    }

    for i, (label, position_col) in axes_map.items():
        position = df[position_col]

        ax = axes[i]

        ax.plot(seconds, position, color='blue', lw = 0.5)

        ax.set_title(label)

    fig.supylabel("position [meters]")
    fig.supxlabel("time [seconds]")

    plt.tight_layout()
    plt.show()

def data_frame_to_figures(df):
    data_frame_accelerations_to_figures(df)
    if ('speed_x' in df.columns):
        data_frame_speeds_to_figures(df)
    if ('position_x' in df.columns):
        data_frame_positions_to_figures(df)

def time_to_str(t):
    minutes = int(t // 60)
    seconds = int(t % 60)

    return f"{minutes}m {seconds}s"

def print_confusion_matrix(matrix, f1_score = -1):
    number_classes = len(matrix)

    print("Confusion Matrix: ")

    sep = "|"
    space = 7
    end = " "

    print(space * " ", end = end)
    for c in range(number_classes):
        if (c == number_classes - 1):
            end = "\n"
        text = "c = " + str(c)
        print(f"{text:^{space}}", end = end)
    
    for l in range(number_classes):
        end = sep
        text = "l = " + str(l)
        print(f"{text:^{space}}", end = end)
        for c in range(number_classes):
            if (c == number_classes - 1):
                end = "\n"
            print(f"{matrix[l][c]:^{space}}", end = end)
    
    print()
    
    n = sum([sum(matrix[c]) for c in range (number_classes)])
    tr = sum([matrix[i][i] for i in range(number_classes)])

    print(f"Accuracy: {tr / n}")
    if (f1_score >= 0):
        print(f"F1-score: {f1_score}")

def confusion_matrix_c(matrix, c):
    number_classes = len(matrix)

    m = [[0, 0], [0, 0]]

    m[0][0] = matrix[c][c]

    m[0][1] = sum([matrix[c][i] for i in range(number_classes)]) - matrix[c][c]

    m[1][0] = sum([matrix[i][c] for i in range(number_classes)]) - matrix[c][c]

    m[1][1] = sum([sum(matrix[c]) for c in range (number_classes)]) - (m[0][0] + m[0][1] + m[1][0])

    return m

def print_confusion_matrix_c(matrix, c):
    print(f"Confusion Matrix {c}: ")

    sep = " "
    space = 20

    print(space * " ", end = sep)
    text = "Predict label " + str(c)
    print(f"{text:^{space}}", end = sep)
    text = "Predict not label " + str(c)
    print(f"{text:^{space}}", end = "\n")
    
    text = "Is label " + str(c)
    print(f"{text:^{space}}", end = sep)
    print(f"{matrix[0][0]:^{space}}", end = sep)
    print(f"{matrix[0][1]:^{space}}", end = "\n")

    text = "Isn't label " + str(c)
    print(f"{text:^{space}}", end = sep)
    print(f"{matrix[1][0]:^{space}}", end = sep)
    print(f"{matrix[1][1]:^{space}}", end = "\n")

    print()
    
    accuracy = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
    precision = 1
    if (matrix[0][0] + matrix[0][1] > 0):
        precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    recall = 1
    if (matrix[0][0] + matrix[1][0] > 0):
        recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])
    f1_score = 2 * precision * recall / (precision + recall)
    
    print(f"Accurary: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1_score}")

    return f1_score

# -------------------------------------------------- continuous Naive Bayes classifier --------------------------------------------------

def gaussian_probability(p, mean, variance):
    assert variance >= 0
    if (variance == 0):
        if (p == mean):
            return 1
        return 0
    return math.exp(- ((p - mean) ** 2) / (2 * variance)) / math.sqrt(2 * math.pi * variance)

def log_gaussian_probability(p, mean, variance):
    assert variance >= 0
    if (variance == 0):
        if (p == mean):
            return 0
        return float("-inf")
    return - (math.log(2 * math.pi * variance) + (((p - mean) ** 2) / variance)) / 2

def update_mle(n, mean, variance, new_p):
    assert variance >= 0
    new_n = n + 1
    new_mean = (n * mean + new_p) / new_n
    new_variance = (n * (variance + (mean ** 2)) + (new_p ** 2)) / new_n - (new_mean ** 2)
    return [new_mean, max(0, new_variance)]

def empty_continuous_Bayesian_classifier(number_classes = 6, number_features = 6, number_seconds = 300):
    classifier = []

    for c in range(number_classes):
        class_c = []
        for f in range(number_features):
            feature = []
            for c in range(number_seconds):
                # initialisation of mean and std
                feature.append([0, 0])
            class_c.append(feature)
        # initialisation of n
        class_c.append(0)
        classifier.append(class_c)
    
    return classifier

def update_continuous_Bayesian_classifier(classifier, df):
    number_classes = len(classifier)
    number_features = len(classifier[0]) - 1
    number_seconds = len(classifier[0][0])
    
    c = df['label'][0]
    assert 0 <= c and c < number_classes
    assert number_features == 6 or number_features == 12
    if (number_features == 6):
        features = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']
    else:
        features = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'speed_x', 'speed_y', 'speed_z', 'position_x', 'position_y', 'position_z']
        df = add_xyz_speed_position(df)

    n = classifier[c][number_features]
    classifier[c][number_features] += 1

    for f in range(number_features):
        for s in range(number_seconds):
            classifier[c][f][s] = update_mle(n, classifier[c][f][s][0], classifier[c][f][s][1], df[features[f]][s])

def train_save_continuous_Bayesian_classifier(file_name, classifier, user = True, start_user_id = 1, end_user_id = 60, start_file_id = 1, end_file_id = 11020, print_i = True):
    start_time = time.time()

    assert 1 <= start_user_id and start_user_id <= TRAIN_USER_NUMBERS
    assert 1 <= end_user_id and end_user_id <= TRAIN_USER_NUMBERS
    assert start_user_id <= end_user_id
    assert 1 <= start_file_id and start_file_id <= TRAIN_FILE_NUMBERS
    assert 1 <= end_file_id and end_file_id <= TRAIN_FILE_NUMBERS
    assert start_file_id <= end_file_id

    if (user):
        for i in range(start_user_id, end_user_id + 1):
            if (print_i and (i - start_user_id + 1) % 6 == 0):
                print(f"train_save_continuous_Bayesian_classifier | user {i}")
                continuous_Bayesian_classifier_to_file(file_name, classifier)
            dfs = user_csv_to_dfs(True, i)
            for df in dfs:
                update_continuous_Bayesian_classifier(classifier, df)
    else:
        for i in range(start_file_id, end_file_id + 1):
            if (print_i and (i - start_file_id + 1) % 1000 == 0):
                print(f"train_save_continuous_Bayesian_classifier | file {i}")
                continuous_Bayesian_classifier_to_file(file_name, classifier)
            df = file_csv_to_df(True, i)
            update_continuous_Bayesian_classifier(classifier, df)
    
    continuous_Bayesian_classifier_to_file(file_name, classifier)

    if (print_i):
        spend_time = time.time() - start_time
        print(f"train_save_continuous_Bayesian_classifier: {time_to_str(spend_time)}")

def imagination_df_continuous_Bayesian_class_c(classifier, c):
    number_classes = len(classifier)
    number_features = len(classifier[0]) - 1
    number_seconds = len(classifier[0][0])
    assert 0 <= c and c < number_classes
    assert number_features == 6 or number_features == 12
    if (number_features == 6):
        features = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']
    else:
        features = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'speed_x', 'speed_y', 'speed_z', 'position_x', 'position_y', 'position_z']

    df = pd.DataFrame()

    df['index'] = [i for i in range(number_seconds)]
    
    for f in range(number_features):
        values = []
        for s in range(number_seconds):
            values.append(classifier[c][f][s][0])
        df[features[f]] = values
    
    return df

def imagination_dfs_continuous_Bayesian_classifier(classifier):
    number_classes = len(classifier)

    dfs = []

    for c in range(number_classes):
        dfs.append(imagination_df_continuous_Bayesian_class_c(classifier, c))
    
    return dfs

def continuous_posterior_class_c(df, classifier, c):
    number_classes = len(classifier)
    number_features = len(classifier[0]) - 1
    number_seconds = len(classifier[0][0])
    assert 0 <= c and c < number_classes
    assert number_features == 6 or number_features == 12
    if (number_features == 6):
        features = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z']
    else:
        features = ['mean_x', 'mean_y', 'mean_z', 'std_x', 'std_y', 'std_z', 'speed_x', 'speed_y', 'speed_z', 'position_x', 'position_y', 'position_z']
        df = add_xyz_speed_position(df)

    # total number of training data
    n = 0
    n_c = classifier[c][number_features]
    
    for i in range(number_classes):
        n += classifier[i][number_features]

    posterior = math.log(n_c / n)
    
    # ??? higher the variance, smaller the error rate
    min_variance = n_c
    for f in range(number_features):
        for s in range(number_seconds):
            value = df[features[f]][s]
            mean = classifier[c][f][s][0]
            variance = classifier[c][f][s][1]

            if (variance == 0):
                variance = min_variance
            
            p = log_gaussian_probability(value, mean, variance)
            posterior += p
    
    return - posterior

def prediction_continuous_Bayesian_classifier(df, classifier, print_i = False):
    number_clases = len(classifier)

    max_c = 0
    posteriors = []
    sum_posteriors = 0

    if (print_i):
        print("Posterior (in log scale):")

    for c in range(number_clases):
        posterior = continuous_posterior_class_c(df, classifier, c)
        posteriors.append(posterior)

        # we inversed the log(posterior) to have a positive value, so smaller the value, bigger the posterior
        if (posterior < posteriors[max_c]):
            max_c = c
        
        sum_posteriors += posterior

    # marginalization
    for c in range(number_clases):
        posteriors[c] /= sum_posteriors
        if (print_i):
            print(str(c) + ": " + str(posteriors[c]))

    return max_c

def predictions_continuous_Bayesian_classifier(file_name, classifier, user = True, start_user_id = 61, end_user_id = 100, start_file_id = 11021, end_file_id = 17869, print_i = True, online = False):
    start_time = time.time()

    assert TRAIN_USER_NUMBERS + 1 <= start_user_id and start_user_id <= TRAIN_USER_NUMBERS + TEST_USER_NUMBERS
    assert TRAIN_USER_NUMBERS + 1 <= end_user_id and end_user_id <= TRAIN_USER_NUMBERS + TEST_USER_NUMBERS
    assert start_user_id <= end_user_id
    assert TRAIN_FILE_NUMBERS + 1 <= start_file_id and start_file_id <= TRAIN_FILE_NUMBERS + TEST_FILE_NUMBERS
    assert TRAIN_FILE_NUMBERS + 1 <= end_file_id and end_file_id <= TRAIN_FILE_NUMBERS + TEST_FILE_NUMBERS
    assert start_file_id <= end_file_id

    rows = []

    if (user):
        for i in range(start_user_id, end_user_id + 1):
            if (print_i and (i - start_user_id + 1) % 6 == 0):
                print(f"predictions_continuous_Bayesian_classifier | user {i}")
            dfs = user_csv_to_dfs(False, i)
            for df in dfs:
                id = df['file_id'][0]
                prediction = prediction_continuous_Bayesian_classifier(df, classifier)
                rows.append({"Id": id, "Label": prediction})
    else:
        for i in range(start_file_id, end_file_id + 1):
            if (print_i and (i - start_file_id + 1) % 1000 == 0):
                print(f"predictions_continuous_Bayesian_classifier | file {i}")
            df = file_csv_to_df(False, i)
            id = df['file_id'][0]
            prediction = prediction_continuous_Bayesian_classifier(df, classifier)
            rows.append({"Id": id, "Label": prediction})
    
    predictions = pd.DataFrame(rows)
    
    predictions.to_csv("baseline_methods\\" + file_name, index = False)
    
    print("\n----------------------------------------------------------------------\n")

    if (print_i):
        spend_time = time.time() - start_time
        print(f"test_continuous_Bayesian_classifier: {time_to_str(spend_time)}")

    return predictions

def test_continuous_Bayesian_classifier(classifier, user = True, start_user_id = 1, end_user_id = 60, start_file_id = 1, end_file_id = 11020, print_i = True, online = False):
    start_time = time.time()

    assert 1 <= start_user_id and start_user_id <= TRAIN_USER_NUMBERS
    assert 1 <= end_user_id and end_user_id <= TRAIN_USER_NUMBERS
    assert start_user_id <= end_user_id
    assert 1 <= start_file_id and start_file_id <= TRAIN_FILE_NUMBERS
    assert 1 <= end_file_id and end_file_id <= TRAIN_FILE_NUMBERS
    assert start_file_id <= end_file_id
    number_classes = len(classifier)
    number_features = len(classifier[0]) - 1

    confusion_matrix = []
    for _ in range(number_classes):
        confusion_matrix.append([0 for _ in range(number_classes)])

    if (user):
        for i in range(start_user_id, end_user_id + 1):
            if (print_i and (i - start_user_id + 1) % 6 == 0):
                print(f"test_continuous_Bayesian_classifier | user {i}")
            imaginations = user_csv_to_dfs(True, i)
            for df in imaginations:
                label = df['label'][0]
                prediction = prediction_continuous_Bayesian_classifier(df, classifier)
                confusion_matrix[label][prediction] += 1
                if (online):
                    update_continuous_Bayesian_classifier(classifier, df)
    else:
        for i in range(start_file_id, end_file_id + 1):
            if (print_i and (i - start_file_id + 1) % 1000 == 0):
                print(f"test_continuous_Bayesian_classifier | file {i}")
            df = file_csv_to_df(True, i)
            label = df['label'][0]
            prediction = prediction_continuous_Bayesian_classifier(df, classifier)
            confusion_matrix[label][prediction] += 1
            if (online):
                update_continuous_Bayesian_classifier(classifier, df)
    
    print("\n----------------------------------------------------------------------\n")

    # F1-score (macro)
    f1_score = 0

    for c in range(number_classes):

        m = confusion_matrix_c(confusion_matrix, c)

        f1_score += print_confusion_matrix_c(m, c)

        print("\n----------------------------------------------------------------------\n")
    
    f1_score /= number_classes
    
    print_confusion_matrix(confusion_matrix, f1_score)

    print("\n----------------------------------------------------------------------\n")

    imaginations = imagination_dfs_continuous_Bayesian_classifier(classifier)

    if (print_i):
        spend_time = time.time() - start_time
        print(f"test_continuous_Bayesian_classifier: {time_to_str(spend_time)}")
    
    for c in range(number_classes):
        data_frame_to_figures(imaginations[c])

# -------------------------------------------------- tests --------------------------------------------------

if __name__ == '__main__':
    # classifier = empty_continuous_Bayesian_classifier(6, 12)

    classifier = file_to_continuous_Bayesian_classifier("acceleration_speed_positions_continuous_Bayesian_classifier.txt")

    # train_save_continuous_Bayesian_classifier("acceleration_speed_positions_continuous_Bayesian_classifier.txt", classifier, True, 1, 60)

    test_continuous_Bayesian_classifier(classifier, True, 1, 60)

    predictions_continuous_Bayesian_classifier("asp_predictions_naive_bayesian_classifier.csv", classifier, True, 61, 100)
