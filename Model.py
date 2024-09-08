

model_mode = "attention"
#model_mode = "unrolling"

if model_mode == "attention":
    from BatchMaker import PfDataVars
    from BatchMaker import PfBatchMaker as BatchMaker
    from OptConfig import OptConfig
    from MotionModel import MotionModelParams
    from AtrappModel import AtrappModel as Model

elif model_mode == "unrolling":
    import Unrolling.graphTools as graphTools
    from Unrolling.UrBatchMaker import UrDataVars
    from Unrolling.UrBatchMaker import UrBatchMaker as BatchMaker
    from Unrolling.UrOptConfig import OptConfig
    from Unrolling.UrMotionModel import UrMotionModelParams as MotionModelParams
    from Unrolling.UnrollingModel import UnrollingModel as Model
    from Unrolling.unrolling_params import N as ur_params_N
    from Unrolling.unrolling_params import M as ur_params_M
    from Unrolling.unrolling_params import graphOptions as graphOptions

