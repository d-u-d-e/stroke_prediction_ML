import numpy as np
from joblib import load


def main():
    # Getting data
    gender = input('Gender (1 = female, 0 = male, -1 = other): ')
    age = input('Age: ')
    ever_married = input('Ever married? (1 = yes, 0 = no): ')
    work_type = input(
        'Work type? (Private: 0, Self-employed: 1, Government Jobs: 2, Work with children: -1, '
        'Never worked: -2): ')
    residence_type = input('Where do you live? (1 = urban zone, 0 = rural zone): ')
    smoking_status = input('Type of smoker? (Unknown: 0, formerly smoked: 1, never smoked: 2, smokes: 3): ')
    hypertension = input('Do you have hypertension problem? (1 = yes, 0 = no): ')
    heart_disease = input('Do you have heart problem? (1 = yes, 0 = no): ')
    avg_glucose_level = input('Avg glucose level: ')
    bmi = input('Bmi parameter: ')

    # Loading model and predict
    rf_model = load('src/saved_models/random_forest.joblib')
    print('Prediction (1 = You are at risk of a stroke, 0 = You are ok): ', rf_model.predict(
        np.array(
            [gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi,
             smoking_status]).reshape(1, -1))[0])


if __name__ == "__main__":
    main()
