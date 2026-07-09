from intake_esgf import ESGFCatalog
import intake_esgf

simus = ['ssp119']
for simu in simus:
    intake_esgf.conf.set(local_cache = [f"/scratch/globc/garcia/scenarIA/rawdata/input4mips/"],
                         num_threads=1
                         )
    intake_esgf.conf.set(indices={"esgf-node.ornl.gov":True})
    print(intake_esgf.conf)
    intake_esgf.conf.save()
    print(intake_esgf.conf['num_threads'])
    cat = ESGFCatalog()
    cat.search(
        institution_id="IACETH",
        frequency='mon'
    )
    #cat.to_dataset_dict()
    #del cat
    '''
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
    '''