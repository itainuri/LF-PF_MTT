
from Unrolling.unrolling_params import thisFilename, SNRstart

if thisFilename == 'particleFilteringSNR': # This is the general name of all related files
    if SNRstart==0:
        ur_state_dict_to_load_str1 = "particleFilteringSNR_0__0.pt" #BESTEST
    elif SNRstart == 2.5:
        ur_state_dict_to_load_str1 = "particleFilteringSNR_2__5.pt" #BEST
    elif SNRstart==5:
        ur_state_dict_to_load_str1 = "particleFilteringSNR_5__0.pt" #BEST
    elif SNRstart == 7.5:
        ur_state_dict_to_load_str1 = "particleFilteringSNR_7__5.pt" #BEST
    elif SNRstart==10:
        ur_state_dict_to_load_str1 = "particleFilteringSNR_10__0.pt" #BEST
    else:
        assert 0, "unrolling_state_dict_str, particleFilteringSNR, no trained weights for SNR"

elif thisFilename == 'particleFilteringNonlinearSNR':
    if SNRstart==0:
        ur_state_dict_to_load_str1 = "particleFilteringNonlinearSNR_0__0.pt" #BESTEST
    elif SNRstart == 2.5:
        ur_state_dict_to_load_str1 = "particleFilteringNonlinearSNR_2__5.pt" #BEST
    elif SNRstart==5:
        ur_state_dict_to_load_str1 = "particleFilteringNonlinearSNR_5__0.pt" #BEST
    elif SNRstart == 7.5:
        ur_state_dict_to_load_str1 = "particleFilteringNonlinearSNR_7__5.pt" #BEST
    elif SNRstart==10:
        ur_state_dict_to_load_str1 = "particleFilteringNonlinearSNR_10__0.pt" #BEST
    else:
        assert 0, "unrolling_state_dict_str, particleFilteringNonlinearSNR, no trained weights for SNR"

elif thisFilename == 'particleFilteringNongaussianSNR':
    if SNRstart==0:
        ur_state_dict_to_load_str1 = "particleFilteringNongaussianSNR_0__0.pt"  #BEST
    elif SNRstart == 2.5:
        ur_state_dict_to_load_str1 = "particleFilteringNongaussianSNR_2__5.pt"  #BEST
    elif SNRstart == 5:
        ur_state_dict_to_load_str1 = "particleFilteringNongaussianSNR_5__0.pt" #BEST
    elif SNRstart == 7.5:
        ur_state_dict_to_load_str1 = "particleFilteringNongaussianSNR_7__5.pt" #BEST
    elif SNRstart==10:
        ur_state_dict_to_load_str1 = "particleFilteringNongaussianSNR_10__0.pt"
    else:
        assert 0, "unrolling_state_dict_str, particleFilteringNongaussianSNR,   no trained weights for SNR"