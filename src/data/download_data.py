from intake_esgf import ESGFCatalog
import intake_esgf

intake_esgf.conf.set(local_cache = ["/scratch/globc/garcia/scenarIA/rawdata/MPI-ESM1-2-LR/mon/ssp126/"])
for i in range(34,51):
    cat = ESGFCatalog()
    cat.search(
        experiment_id="ssp126",
        source_id="MPI-ESM1-2-LR",
        variable_id=["tas", "pr"],
        frequency="day",
        table_id="day",
        member_id =f'r{i}i1p1f1'
    )
    cat.to_dataset_dict()