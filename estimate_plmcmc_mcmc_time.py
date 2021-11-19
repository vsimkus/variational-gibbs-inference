import json
import numpy as np

from cdi.util.utils import EpochScheduler, EpochIntervalScheduler

path_template = "experiment_configs/flows_uci/learning_experiments/{}/{}.json"

# exp = 'rqcspline_miniboone_plmcmc'
# time_per_imp_step = np.array([750,766,750])/500
# time_per_step = np.array([32,32,32])  # Approx

# exp = 'rqcspline_gas_plmcmc'
# time_per_imp_step = np.array([12300, 12744, 9600])/500
# # time_per_step = np.array([910, 960, 740])  # OLD with wrong batch_size
# time_per_step = np.array([500,480,250])  # Approx

exp = 'rqcspline_power_plmcmc'
time_per_imp_step = np.array([16331,21024,14600])/500
# time_per_step = np.array([1860,1780,1250])  # OLD with wrong batch_size
time_per_step = np.array([1060,890,450])  # Approx

# exp = 'rqcspline_hepmass_plmcmc'
# time_per_imp_step = np.array([10500,10650,10500])/500
# # time_per_step = np.array([608,608,608])  # OLD with wrong batch_size
# time_per_step = np.array([350,350,350])  # Approx


for i, g in enumerate((1, 3, 5)):
    path = path_template.format(g, exp)

    with open(path) as f:
        # Strip line comments before parsing json
        config = ''.join(line for line in f if not line.lstrip().startswith('//'))
        config = json.loads(config)

    epochs = config['max_epochs']

    num_imp_steps_schedule = EpochScheduler(
                        None,
                        config['plmcmc']['num_imp_steps_schedule'],
                        config['plmcmc']['num_imp_steps_schedule_values'])

    update_imputations_schedule = EpochIntervalScheduler(
                        None,
                        config['plmcmc']['update_imputations_schedule_init_value'],
                        config['plmcmc']['update_imputations_schedule_main_value'],
                        config['plmcmc']['update_imputations_schedule_other_value'],
                        config['plmcmc']['update_imputations_schedule_start_epoch'],
                        config['plmcmc']['update_imputations_schedule_period'])

    total = 0
    times_updated = 0
    for e in range(epochs):
        imp_steps = num_imp_steps_schedule.get_value(e)
        update_imps = update_imputations_schedule.get_value(e)
        if update_imps:
            times_updated += 1
            total += imp_steps

    print(f'---- {g/6:.2f}% missingness ----')
    print('Times imps updated', times_updated)
    print('Est. MCMC time', f'{total*time_per_imp_step[i]/60/60}h')
    print('Est. MCMC time', f'{total*time_per_imp_step[i]/60/60/24}d')
    print('Est. fitting time', f'{epochs*time_per_step[i]/60/60}h')
    print('Est. fitting time', f'{epochs*time_per_step[i]/60/60/24}d')
    print('Est. total time', f'{(total*time_per_imp_step+epochs*time_per_step)[i]/60/60}h')
    print('Est. total time', f'{(total*time_per_imp_step+epochs*time_per_step)[i]/60/60/24}d')
