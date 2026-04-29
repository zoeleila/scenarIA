from intake_esgf import ESGFCatalog
import intake_esgf
simus = ['hist-aer', 'hist-nat', 'hist-GHG']
for simu in simus:
    intake_esgf.conf.set(local_cache = [f"/scratch/globc/garcia/scenarIA/rawdata/MPI-ESM1-2-LR/mon/{simu}/"],
                         num_threads=12
                         )
    print(intake_esgf.conf['num_threads'])
    for i in range(1,31):
        print(i)
        cat = ESGFCatalog()
        cat.search(
            experiment_id=simu,
            source_id="MPI-ESM1-2-LR",
            variable_id=["tas", "pr"],
            table_id='Amon',
            member_id =f'r{i}i1p1f1'
        )
        cat.to_dataset_dict()
        del cat
