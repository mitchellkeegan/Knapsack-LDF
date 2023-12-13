from ML_Model_Defs import RF_model, NN_model, ANN

directories = {'Instance Type': 'Weights_and_Values_and_Capacity',
               'problem': '0-1 Knapsack',
               'Opt Model': 'Vanilla'}

model_params = {'Batch Norm': True,
                'Dropout': False}

training_params = {'lr': 1e-4,
                   'Epochs': 500,
                   'Training Batch Size': 256,
                   'Lagrange Step': 1e-7,
                   'Initial Lagrange Multiplier': [[1.]],
                   'k Round': 25,
                   'Constraints': 'LDF',
                   'Clip Grad Norm': True,
                   'Max Grad Norm': 0.5,
                   'Grid Search': True,
                   'LM Step Scheduler': None,
                   'ctype': 'violation',
                   'LM Delay': 0,
                   'Lambda Norms': [0,0.5,1,2,4,6,8,10]}

directory_number = 1

for lr in [1e-4]:
    training_params['lr'] = lr
    for ls in [1e-7]:
        training_params['Lagrange Step'] = ls
        for grad_norm in [0.5]:
            training_params['Max Grad Norm'] = grad_norm
            directories['Hyperparameters'] = f'GN: {grad_norm}'
            directory_number += 1

            model = NN_model(model_params, training_params, directories, ANN)
            model.load_data()
            model.fit()