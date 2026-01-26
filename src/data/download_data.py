from intake_esgf import ESGFCatalog
import intake_esgf
simus = ['ssp370', 'ssp585', 'historical']
for simu in simus:
    intake_esgf.conf.set(local_cache = [f"/scratch/globc/garcia/scenarIA/rawdata/MPI-ESM1-2-LR/mon/{simu}/"])
    if simu == 'ssp370':
        start = 38
    else:
        start = 30
    for i in range(start,51):
        print(i)
        cat = ESGFCatalog()
        cat.search(
            experiment_id=simu,
            source_id="MPI-ESM1-2-LR",
            variable_id=["tas", "pr"],
            frequency="day",
            table_id="day",
            member_id =f'r{i}i1p1f1'
        )
        cat.to_dataset_dict()
        del cat
