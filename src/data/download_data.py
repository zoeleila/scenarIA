from intake_esgf import ESGFCatalog
import intake_esgf
simus = ['historical']
for simu in simus:
    intake_esgf.conf.set(local_cache = [f"/scratch/globc/garcia/scenarIA/rawdata/MPI-ESM1-2-LR/mon/{simu}/"],
                         num_threads=12
                         )
    print(intake_esgf.conf['num_threads'])
    start = 42
    tableid = 'day'
    for i in range(start,51):
        print(i)
        cat = ESGFCatalog()
        cat.search(
            experiment_id=simu,
            source_id="MPI-ESM1-2-LR",
            variable_id=["tas", "pr"],
            table_id=tableid,
            member_id =f'r{i}i1p1f1'
        )
        cat.to_dataset_dict()
        del cat
