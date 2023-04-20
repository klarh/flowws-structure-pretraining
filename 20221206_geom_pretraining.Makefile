# Determine this makefile's path.
THIS_FILE := $(MAKEFILE_LIST)

DATA_DIR:=pyriodic-aflow
#DATA_DIR:=maya
ZOOMED_DATA_DIR:=${DATA_DIR}

NEIGHBOR_MODE:=nearest

ifeq ($(NEIGHBOR_MODE),sann)
	NEIGHBOR_CALC:=SANNeighbors
else ifeq ($(NEIGHBOR_MODE),rdfdist1)
	NEIGHBOR_CALC:=DistanceNeighbors --rdf-shells 1 --max-neighbors 32
else ifeq ($(NEIGHBOR_MODE),rdfdist2)
	NEIGHBOR_CALC:=DistanceNeighbors --rdf-shells 2 --max-neighbors 64
else ifeq ($(NEIGHBOR_MODE),nearest)
	NEIGHBOR_COUNT:=20
	NEIGHBOR_CALC:=NearestNeighbors --neighbor-count ${NEIGHBOR_COUNT}
endif

PYTHON:=python
SEED:=13
EPOCHS:=128
MULTILABEL:=0
VAE_SCALE:=1e-5
PYRIODIC_SIZE:=4096
PYRIODIC_NOISE:=1e-2 3e-2 5e-2
FRAME_SKIP:=1
MOVIE_OP_THRESH:=
MOVIE_FRAMERATE:=24
EXTRA_TRAIN_ARGS:=
DEPTH:=3
WIDTH:=32
DENOISING_NOISE:=0.5
VAL_SPLIT:=0.3

INITIALIZETF_EXTRA_ARGS:=
FRAME_CLASSIFICATION_EXTRA_ARGS:=
TRANSFER_SUBSAMPLE:=1e-3
NET_DISTANCE_NORMALIZATION:=none
SELF_CLASSIFICATION_NOISE:=0.25

# data
# fcc bcc hcp
# e8ec tP30
# c1dba clathrate
# d57c iQc
STRUC_FLUID_FRAMES:=FileLoader -f ${DATA_DIR}/struc_prediction/workspace/ff1e2b1da3eccc022eca819dd7108606/dump.gsd --frame-start 46 --frame-end 47 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/81a6c5396a501f7e939001ffff307462/dump.gsd --frame-start 72 --frame-end 73 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/583e02d7d1ae5e2bba602747e2bf034d/dump.gsd --frame-start 44 --frame-end 45 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/e8ecb61993395e7ced88692e33fc3932/dump.gsd --frame-start 32 --frame-end 33 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/c1dba125a3784fd979a4d0f53e7959a9/dump.gsd --frame-start 27 --frame-end 28 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/d57c4440487f5feb1415a902934e49e7/dump.gsd --frame-start 11 --frame-end 12
STRUC_FINAL_FRAMES:=FileLoader -f ${DATA_DIR}/struc_prediction/workspace/ff1e2b1da3eccc022eca819dd7108606/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/81a6c5396a501f7e939001ffff307462/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/583e02d7d1ae5e2bba602747e2bf034d/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/e8ecb61993395e7ced88692e33fc3932/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/c1dba125a3784fd979a4d0f53e7959a9/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/d57c4440487f5feb1415a902934e49e7/dump.gsd --frame-start -1
STRUC_FRAMES:=${STRUC_FLUID_FRAMES} ${STRUC_FINAL_FRAMES}
FINAL_FRAMES:=FileLoader -f $$(find ${DATA_DIR} -name dump.gsd | sort) --frame-start -1
INTERMEDIATE_FRAMES:=FileLoader -f $$(find ${DATA_DIR} -name dump.gsd | sort) --frame-start 4 --frame-skip 20
PYRIODIC_TRAIN_STRUCTURES:=cF4-Cu cI2-W cP1-Po cP46-Si hP2-Mg tP30-U
PYRIODIC_TRAIN_FRAMES:=PyriodicLoader -z ${PYRIODIC_SIZE} -s ${PYRIODIC_TRAIN_STRUCTURES} -n ${PYRIODIC_NOISE}
PYRIODIC_NOTRAIN_STRUCTURES:=aP4-Cf cF136-Si cF8-C cI16-Li cI16-Si cI58-Mn cP20-Mn hP1-HgSn6-10 hP3-Se hP4-La hP6-C hP6-Sc hR1-Hg hR1-Po hR105-B hR12-B hR2-As hR2-C hR3-Sm mC12-Po mC16-C mC34-Pu mP16-Pu mP32-Se mP4-Te mP8-C mP84-P oC4-U oC8-Ga oC8-P oF128-S oF8-Pu oP16-C oP8-Np tI16-S tI2-In tI2-Pa tI4-Si tI4-Sn tI8-C tP12-C tP12-Si tP4-Np tP50-B
PYRIODIC_NOTRAIN_FRAMES:=PyriodicLoader -z ${PYRIODIC_SIZE} -s ${PYRIODIC_NOTRAIN_STRUCTURES} -n ${PYRIODIC_NOISE}
PYRIODIC_ALL_STRUCTURES:=${PYRIODIC_NOTRAIN_STRUCTURES} ${PYRIODIC_TRAIN_STRUCTURES}
PYRIODIC_ALL_FRAMES:=PyriodicLoader -z ${PYRIODIC_SIZE} -s ${PYRIODIC_ALL_STRUCTURES} -n ${PYRIODIC_NOISE}
#GEOM_ALL_FRAMES:=GEOMLoader -d geom_json
GEOM_ALL_FRAMES:=GEOMLoader -d /scratch/ssd002/datasets/GEOM/extracted_geom/

# tasks
NOISY_BOND:=NoisyBondTask -n .5 --seed ${SEED}
PROPERTY_NOISY_BOND:=PropertyNoisyBondTask --noise-magnitude 0.1 --seed ${SEED}
NEAREST_BOND:=NearestBondTask --seed ${SEED}
AUTOENCODER:=AutoencoderTask --seed ${SEED}
DENOISING:=DenoisingTask --noise ${DENOISING_NOISE} --seed ${SEED} --register 1
SHIFT_ID:=ShiftIdentificationTask --scale .5 --seed ${SEED}
FRAME_CLASSIFICATION:=FrameClassificationTask --seed ${SEED} --multilabel ${MULTILABEL} --per-cloud 1 ${FRAME_CLASSIFICATION_EXTRA_ARGS}
FRAME_REGRESSION:=FrameRegressionTask --seed ${SEED}
GEOM_REGRESSION:=GEOMRegressionTask --seed ${SEED}
#GEOM_REGRESSION:=GalaPotentialRegressor --predict-energy 0 --predict-forces 1 --seed ${SEED}

# models
#NOISY_BOND_ARCH:=GalaBondClassifier --num-blocks ${DEPTH} --n-dim ${WIDTH} --use-multivectors 1 --invar-mode full --covar-mode full --dropout .5 --equivariant-value-normalization momentum_layer --include-normalized-products 1 --normalize-equivariant-values 1 --normalize-distances ${NET_DISTANCE_NORMALIZATION}
NOISY_BOND_ARCH:=GalaBondClassifier --num-blocks ${DEPTH} --n-dim ${WIDTH} --use-multivectors 1 --invar-mode full --covar-mode full --dropout .5 --equivariant-value-normalization momentum_layer --include-normalized-products 0 --normalize-equivariant-values 1 --normalize-distances ${NET_DISTANCE_NORMALIZATION}
NEAREST_BOND_ARCH:=GalaBondRegressor --num-blocks ${DEPTH} --n-dim ${WIDTH} --use-multivectors 1 --invar-mode full --covar-mode full --drop-geometric-embeddings 1 --dropout 1e-1 --equivariant-value-normalization none --include-normalized-products 1 --normalize-equivariant-values 1 --normalize-distances ${NET_DISTANCE_NORMALIZATION}
AUTOENCODER_ARCH:=GalaBottleneckAutoencoder --num-blocks ${DEPTH} --n-dim ${WIDTH} --num-vector-blocks 0 --use-multivectors 1 --cross-attention 0 --invar-mode full --covar-mode full --vae-scale ${VAE_SCALE} --dropout 1e-1 --equivariant-value-normalization none --include-normalized-products 1 --normalize-equivariant-values 1 --normalize-distances ${NET_DISTANCE_NORMALIZATION}
DENOISING_ARCH:=GalaVectorAutoencoder --num-blocks ${DEPTH} --n-dim ${WIDTH} --num-vector-blocks 0 --dropout 1e-1 --use-multivectors 1 --invar-mode full --covar-mode full --drop-geometric-embeddings 1 --equivariant-value-normalization none --include-normalized-products 1 --normalize-equivariant-values 1 --normalize-distances ${NET_DISTANCE_NORMALIZATION}
SHIFT_ID_ARCH:=GalaBondRegressor --num-blocks ${DEPTH} --n-dim ${WIDTH} --use-multivectors 1 --invar-mode full --covar-mode full --drop-geometric-embeddings 1 --dropout 1e-1 --equivariant-value-normalization none --include-normalized-products 1 --normalize-equivariant-values 1 --normalize-distances ${NET_DISTANCE_NORMALIZATION}
FRAME_CLASSIFICATION_ARCH:=GalaBondClassifier --num-blocks ${DEPTH} --n-dim ${WIDTH} --use-multivectors 1 --invar-mode full --covar-mode full --merge-fun concat --join-fun concat --dropout .5 --equivariant-value-normalization momentum_layer --include-normalized-products 1 --normalize-equivariant-values 1 --normalize-distances ${NET_DISTANCE_NORMALIZATION}
FRAME_REGRESSION_ARCH:=GalaScalarRegressor --num-blocks ${DEPTH} --n-dim ${WIDTH} --use-multivectors 1 --invar-mode full --covar-mode full --dropout .1 --equivariant-value-normalization momentum_layer --include-normalized-products 1 --normalize-equivariant-values 1 --normalize-distances ${NET_DISTANCE_NORMALIZATION}
GEOM_REGRESSION_ARCH:=GalaPotentialRegressor --predict-energy 1 --predict-forces 0 --num-blocks ${DEPTH} --n-dim ${WIDTH} --use-multivectors 1 --invar-mode full --covar-mode full --dropout 0 --include-normalized-products 0 --normalize-equivariant-values 0 --normalize-distances ${NET_DISTANCE_NORMALIZATION}
#GEOM_REGRESSION_ARCH:=GalaPotentialRegressor --predict-energy 1 --predict-forces 0 --num-blocks ${DEPTH} --n-dim ${WIDTH} --use-multivectors 1 --invar-mode full --covar-mode full --dropout .1 --equivariant-value-normalization momentum_layer --include-normalized-products 0 --normalize-equivariant-values 1 --normalize-distances ${NET_DISTANCE_NORMALIZATION}

# training
#GENERATOR_TRAIN:=flowws_keras_experimental.Train --epochs ${EPOCHS} --generator-train-steps 2048 --generator-val-steps 512 --use-multiprocessing False --catch-keyboard-interrupt 1 --reduce-lr 4 --early-stopping 10 --early-stopping-best 1 --recompile 1 --accumulate-gradients 8 --seed ${SEED} ${EXTRA_TRAIN_ARGS}
GENERATOR_TRAIN:=flowws_keras_experimental.Train --epochs 50 --generator-train-steps 2048 --generator-val-steps 512 --use-multiprocessing False --catch-keyboard-interrupt 1 --reduce-lr 50 --early-stopping 50 --early-stopping-best 1 --recompile 1  --seed ${SEED} ${EXTRA_TRAIN_ARGS}
#STATIC_TRAIN:=flowws_keras_experimental.Train --epochs ${EPOCHS} --validation-split ${VAL_SPLIT} --batch-size 4 --use-multiprocessing False --catch-keyboard-interrupt 1 --reduce-lr 1 --early-stopping 2 --early-stopping-best 1 --recompile 1 --accumulate-gradients 8 --seed ${SEED} ${EXTRA_TRAIN_ARGS}
STATIC_TRAIN:=flowws_keras_experimental.Train --epochs 10000 --validation-split ${VAL_SPLIT} --batch-size 4 --use-multiprocessing False --catch-keyboard-interrupt 1 --reduce-lr 2500 --early-stopping 2500 --early-stopping-best 1 --recompile 1  --seed ${SEED} ${EXTRA_TRAIN_ARGS}

NOISY_BOND_TRAIN:=${GENERATOR_TRAIN}
NEAREST_BOND_TRAIN:=${STATIC_TRAIN}
AUTOENCODER_TRAIN:=${STATIC_TRAIN}
DENOISING_TRAIN:=${GENERATOR_TRAIN}
SHIFT_ID_TRAIN:=${GENERATOR_TRAIN}
FRAME_CLASSIFICATION_TRAIN:=${STATIC_TRAIN}
FRAME_REGRESSION_TRAIN:=${STATIC_TRAIN}
GEOM_REGRESSION_TRAIN:=${STATIC_TRAIN}

.PHONY: all
all: all_final_frames all_intermediate_frames all_struc_frames

.PHONY: all_final_frames
all_final_frames: dump.noisy_bond.final_frames.sqlite dump.nearest_bond.final_frames.sqlite dump.autoencoder.final_frames.sqlite dump.denoising.final_frames.sqlite dump.shift_identification.final_frames.sqlite dump.frame_classification.final_frames.sqlite

dump.noisy_bond.final_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${FINAL_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${NOISY_BOND} ${NOISY_BOND_ARCH} ${NOISY_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f noisy_bond final_frames

dump.nearest_bond.final_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${FINAL_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${NEAREST_BOND} ${NEAREST_BOND_ARCH} ${NEAREST_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f nearest_bond final_frames

dump.autoencoder.final_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${FINAL_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${AUTOENCODER} ${AUTOENCODER_ARCH} ${AUTOENCODER_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f autoencoder final_frames

dump.denoising.final_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${FINAL_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${DENOISING} ${DENOISING_ARCH} ${DENOISING_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f denoising final_frames

dump.shift_identification.final_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${FINAL_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${SHIFT_ID} ${SHIFT_ID_ARCH} ${SHIFT_ID_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f shift_identification final_frames

dump.frame_classification.final_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${FINAL_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${FRAME_CLASSIFICATION} ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f frame_classification final_frames

.PHONY: all_intermediate_frames
all_intermediate_frames: dump.noisy_bond.intermediate_frames.sqlite dump.nearest_bond.intermediate_frames.sqlite dump.autoencoder.intermediate_frames.sqlite dump.denoising.intermediate_frames.sqlite dump.shift_identification.intermediate_frames.sqlite dump.frame_classification.intermediate_frames.sqlite

dump.noisy_bond.intermediate_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${INTERMEDIATE_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${NOISY_BOND} --subsample .01 ${NOISY_BOND_ARCH} ${NOISY_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f noisy_bond intermediate_frames

dump.nearest_bond.intermediate_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${INTERMEDIATE_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${NEAREST_BOND} --subsample .01 ${NEAREST_BOND_ARCH} ${NEAREST_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f nearest_bond intermediate_frames

dump.autoencoder.intermediate_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${INTERMEDIATE_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${AUTOENCODER} --subsample .01 ${AUTOENCODER_ARCH} ${AUTOENCODER_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f autoencoder intermediate_frames

dump.denoising.intermediate_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${INTERMEDIATE_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${DENOISING} --subsample .01 ${DENOISING_ARCH} ${DENOISING_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f denoising intermediate_frames

dump.shift_identification.intermediate_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${INTERMEDIATE_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${SHIFT_ID} --subsample .01 ${SHIFT_ID_ARCH} ${SHIFT_ID_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f shift_identification intermediate_frames

dump.frame_classification.intermediate_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${INTERMEDIATE_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${FRAME_CLASSIFICATION} --subsample .01 ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f frame_classification intermediate_frames

.PHONY: all_struc_frames
all_struc_frames: dump.noisy_bond.struc_frames.sqlite dump.nearest_bond.struc_frames.sqlite dump.autoencoder.struc_frames.sqlite dump.denoising.struc_frames.sqlite dump.shift_identification.struc_frames.sqlite dump.frame_classification.struc_frames.sqlite

dump.noisy_bond.struc_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${STRUC_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${NOISY_BOND} ${NOISY_BOND_ARCH} ${NOISY_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f noisy_bond struc_frames

dump.nearest_bond.struc_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${STRUC_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${NEAREST_BOND} ${NEAREST_BOND_ARCH} ${NEAREST_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f nearest_bond struc_frames

dump.autoencoder.struc_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${STRUC_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${AUTOENCODER} ${AUTOENCODER_ARCH} ${AUTOENCODER_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f autoencoder struc_frames

dump.denoising.struc_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${STRUC_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${DENOISING} ${DENOISING_ARCH} ${DENOISING_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f denoising struc_frames

dump.shift_identification.struc_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${STRUC_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${SHIFT_ID} ${SHIFT_ID_ARCH} ${SHIFT_ID_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f shift_identification struc_frames

dump.frame_classification.struc_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${STRUC_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${FRAME_CLASSIFICATION} ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f frame_classification struc_frames

dump.frame_classification.fluid_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${STRUC_FLUID_FRAMES} \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${FRAME_CLASSIFICATION} ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f frame_classification fluid_frames

output.ClassifierPlotter.frame_classification.fluid_frames.cF4_Cu.pdf: dump.frame_classification.fluid_frames.sqlite
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/ff1e2b1da3eccc022eca819dd7108606/dump.gsd --frame-start 4 --frame-skip 4 \
		LoadModel --filename $< --subsample 1e-1 \
		ClassifierPlotter -s 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f frame_classification fluid_frames cF4_Cu

output.ClassifierPlotter.frame_classification.fluid_frames.cI2_W.pdf: dump.frame_classification.fluid_frames.sqlite
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/81a6c5396a501f7e939001ffff307462/dump.gsd --frame-start 4 --frame-skip 4 \
		LoadModel --filename $< --subsample 1e-1 \
		ClassifierPlotter -s 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f frame_classification fluid_frames cI2_W

output.ClassifierPlotter.frame_classification.fluid_frames.hP2_Mg.pdf: dump.frame_classification.fluid_frames.sqlite
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/583e02d7d1ae5e2bba602747e2bf034d/dump.gsd --frame-start 4 --frame-skip 4 \
		LoadModel --filename $< --subsample 1e-1 \
		ClassifierPlotter -s 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f frame_classification fluid_frames hP2_Mg

output.ClassifierPlotter.frame_classification.fluid_frames.tP30_CrFe.pdf: dump.frame_classification.fluid_frames.sqlite
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/e8ecb61993395e7ced88692e33fc3932/dump.gsd --frame-start 4 --frame-skip 4 \
		LoadModel --filename $< --subsample 1e-1 \
		ClassifierPlotter -s 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f frame_classification fluid_frames tP30_CrFe

.PHONY: all_pyriodic_notrain_frames
all_pyriodic_notrain_frames: dump.noisy_bond.pyriodic_notrain_frames.sqlite dump.nearest_bond.pyriodic_notrain_frames.sqlite dump.autoencoder.pyriodic_notrain_frames.sqlite dump.denoising.pyriodic_notrain_frames.sqlite dump.shift_identification.pyriodic_notrain_frames.sqlite dump.frame_classification.pyriodic_notrain_frames.sqlite

dump.noisy_bond.pyriodic_notrain_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_NOTRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${NOISY_BOND} ${NOISY_BOND_ARCH} ${NOISY_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f noisy_bond pyriodic_notrain_frames

dump.nearest_bond.pyriodic_notrain_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_NOTRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${NEAREST_BOND} ${NEAREST_BOND_ARCH} ${NEAREST_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f nearest_bond pyriodic_notrain_frames

dump.autoencoder.pyriodic_notrain_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_NOTRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${AUTOENCODER} ${AUTOENCODER_ARCH} ${AUTOENCODER_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f autoencoder pyriodic_notrain_frames

dump.denoising.pyriodic_notrain_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_NOTRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${DENOISING} ${DENOISING_ARCH} ${DENOISING_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f denoising pyriodic_notrain_frames

dump.shift_identification.pyriodic_notrain_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_NOTRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${SHIFT_ID} ${SHIFT_ID_ARCH} ${SHIFT_ID_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f shift_identification pyriodic_notrain_frames

dump.frame_classification.pyriodic_notrain_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_NOTRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${FRAME_CLASSIFICATION} ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f frame_classification pyriodic_notrain_frames

.PHONY: all_pyriodic_train_frames
all_pyriodic_train_frames: dump.noisy_bond.pyriodic_train_frames.sqlite dump.nearest_bond.pyriodic_train_frames.sqlite dump.autoencoder.pyriodic_train_frames.sqlite dump.denoising.pyriodic_train_frames.sqlite dump.shift_identification.pyriodic_train_frames.sqlite dump.frame_classification.pyriodic_train_frames.sqlite

dump.noisy_bond.pyriodic_train_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_TRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${NOISY_BOND} ${NOISY_BOND_ARCH} ${NOISY_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f noisy_bond pyriodic_train_frames

dump.nearest_bond.pyriodic_train_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_TRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${NEAREST_BOND} ${NEAREST_BOND_ARCH} ${NEAREST_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f nearest_bond pyriodic_train_frames

dump.autoencoder.pyriodic_train_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_TRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${AUTOENCODER} ${AUTOENCODER_ARCH} ${AUTOENCODER_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f autoencoder pyriodic_train_frames

dump.denoising.pyriodic_train_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_TRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${DENOISING} ${DENOISING_ARCH} ${DENOISING_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f denoising pyriodic_train_frames

dump.shift_identification.pyriodic_train_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_TRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${SHIFT_ID} ${SHIFT_ID_ARCH} ${SHIFT_ID_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f shift_identification pyriodic_train_frames

dump.frame_classification.pyriodic_train_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_TRAIN_FRAMES} \
		${NEIGHBOR_CALC} \
		${FRAME_CLASSIFICATION} ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f frame_classification pyriodic_train_frames

.PHONY: all_pyriodic_all_frames
all_pyriodic_all_frames: dump.noisy_bond.pyriodic_all_frames.sqlite dump.nearest_bond.pyriodic_all_frames.sqlite dump.autoencoder.pyriodic_all_frames.sqlite dump.denoising.pyriodic_all_frames.sqlite dump.shift_identification.pyriodic_all_frames.sqlite dump.frame_classification.pyriodic_all_frames.sqlite

dump.noisy_bond.pyriodic_all_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_ALL_FRAMES} \
		${NEIGHBOR_CALC} \
		${NOISY_BOND} ${NOISY_BOND_ARCH} ${NOISY_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f noisy_bond pyriodic_all_frames

dump.property_noisy_bond.geom_all_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${GEOM_ALL_FRAMES} \
		${NEIGHBOR_CALC} \
		${PROPERTY_NOISY_BOND} ${NOISY_BOND_ARCH} ${NOISY_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f noisy_bond geom_all_frames_random_50e

dump.nearest_bond.pyriodic_all_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_ALL_FRAMES} \
		${NEIGHBOR_CALC} \
		${NEAREST_BOND} --subsample .1 ${NEAREST_BOND_ARCH} ${NEAREST_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f nearest_bond pyriodic_all_frames

dump.autoencoder.pyriodic_all_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_ALL_FRAMES} \
		${NEIGHBOR_CALC} \
		${AUTOENCODER} --subsample .1 ${AUTOENCODER_ARCH} ${AUTOENCODER_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f autoencoder pyriodic_all_frames

dump.denoising.pyriodic_all_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_ALL_FRAMES} \
		${NEIGHBOR_CALC} \
		${DENOISING} ${DENOISING_ARCH} ${DENOISING_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f denoising pyriodic_all_frames

dump.shift_identification.pyriodic_all_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_ALL_FRAMES} \
		${NEIGHBOR_CALC} \
		${SHIFT_ID} ${SHIFT_ID_ARCH} ${SHIFT_ID_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f shift_identification pyriodic_all_frames

dump.frame_classification.pyriodic_all_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_ALL_FRAMES} \
		${NEIGHBOR_CALC} \
		${FRAME_CLASSIFICATION} --subsample .1 ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f frame_classification pyriodic_all_frames
	
#dump.frame_regression.geom_all_frames.sqlite: dump.noisy_bond.geom_all_frames.sqlite
#LoadModel --filename $< --only-model 1 
	#${NEIGHBOR_CALC} 
dump.frame_regression.geom_all_frames.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${GEOM_ALL_FRAMES} -v ${VAL_SPLIT} \
	    ${NEIGHBOR_CALC} \
		${GEOM_REGRESSION} ${GEOM_REGRESSION_ARCH} ${GEOM_REGRESSION_TRAIN} --shuffle False \
		flowws_keras_experimental.Save --save-model 1 -f frame_regression geom_all_frames_pretrain_md17_200e


output.EmbeddingPlotter.structure_frames_with_iQc.%.pca.pdf: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/ff1e2b1da3eccc022eca819dd7108606/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/81a6c5396a501f7e939001ffff307462/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/583e02d7d1ae5e2bba602747e2bf034d/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/e8ecb61993395e7ced88692e33fc3932/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/c1dba125a3784fd979a4d0f53e7959a9/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/d57c4440487f5feb1415a902934e49e7/dump.gsd --frame-start 4 --frame-end 21 --frame-skip 4 \
		LoadModel -f "$<" --subsample .1 \
		EvaluateEmbedding --average-bonds 1 \
		PCAEmbedding \
		EmbeddingPlotter -r 1 --shuffle 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f structure_frames_with_iQc "$<" pca

output.EmbeddingPlotter.structure_frames_with_iQc.%.umap.pdf: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/ff1e2b1da3eccc022eca819dd7108606/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/81a6c5396a501f7e939001ffff307462/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/583e02d7d1ae5e2bba602747e2bf034d/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/e8ecb61993395e7ced88692e33fc3932/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/c1dba125a3784fd979a4d0f53e7959a9/dump.gsd --frame-start -1 FileLoader -f ${DATA_DIR}/struc_prediction/workspace/d57c4440487f5feb1415a902934e49e7/dump.gsd --frame-start 4 --frame-end 21 --frame-skip 4 \
		LoadModel -f "$<" --subsample .1 \
		EvaluateEmbedding --average-bonds 1 \
		UMAPEmbedding \
		EmbeddingPlotter -r 1 --shuffle 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f structure_frames_with_iQc "$<" umap

dump.pyriodic_distance_embedding.cP1_Po.%.zip: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_ALL_FRAMES} \
		LoadModel --filename $< --subsample 1e-2 \
		EvaluateEmbedding \
		EmbeddingDistanceTrajectory -i ${ZOOMED_DATA_DIR}/ae0780dec9c965ec0d7275b05153d380/dump.gsd --frame-start 54580 --frame-end 55631 --frame-skip ${FRAME_SKIP} -m log_nearest -c percentile -o "$@" ${DISTANCE_EMBEDDING_ARGS}

dump.pyriodic_distance_embedding.cI2_W.%.zip: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_ALL_FRAMES} \
		LoadModel --filename $< --subsample 1e-2 \
		EvaluateEmbedding \
		EmbeddingDistanceTrajectory -i ${ZOOMED_DATA_DIR}/5761865812ffaad6df8120bf3f7a68a8/dump.gsd --frame-start 32000 --frame-end 32401 --frame-skip ${FRAME_SKIP} -m log_nearest -c percentile -o "$@" ${DISTANCE_EMBEDDING_ARGS}

dump.pyriodic_distance_embedding.iQc.%.zip: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		${PYRIODIC_ALL_FRAMES} \
		LoadModel --filename $< --subsample 1e-2 \
		EvaluateEmbedding \
		EmbeddingDistanceTrajectory -i ${ZOOMED_DATA_DIR}/3aa147fbdfff5f5da0c3a84a36b1ab89/dump.gsd --frame-start 13000 --frame-end 16301 --frame-skip ${FRAME_SKIP} -m log_nearest -c percentile -o "$@" ${DISTANCE_EMBEDDING_ARGS}

output.EmbeddingPlotter.iQc.%.pca.pdf: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${ZOOMED_DATA_DIR}/3aa147fbdfff5f5da0c3a84a36b1ab89/dump.gsd --frame-start 13000 --frame-end 16301 --frame-skip 8 \
		LoadModel -f "$<" --subsample 5e-4 \
		EvaluateEmbedding --average-bonds 0 \
		PCAEmbedding \
		EmbeddingPlotter -r 1 --shuffle 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f iQc "$<" pca

output.EmbeddingPlotter.iQc.%.umap.pdf: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${ZOOMED_DATA_DIR}/3aa147fbdfff5f5da0c3a84a36b1ab89/dump.gsd --frame-start 13000 --frame-end 16301 --frame-skip 8 \
		LoadModel -f "$<" --subsample 5e-4 \
		EvaluateEmbedding --average-bonds 0 \
		UMAPEmbedding \
		EmbeddingPlotter -r 1 --shuffle 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f iQc "$<" umap

output.EmbeddingPlotter.cI2_W.%.pca.pdf: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${ZOOMED_DATA_DIR}/5761865812ffaad6df8120bf3f7a68a8/dump.gsd --frame-start 32000 --frame-end 32401 \
		LoadModel -f "$<" --subsample 5e-4 \
		EvaluateEmbedding --average-bonds 0 \
		PCAEmbedding \
		EmbeddingPlotter -r 1 --shuffle 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f cI2_W "$<" pca

output.EmbeddingPlotter.cI2_W.%.umap.pdf: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${ZOOMED_DATA_DIR}/5761865812ffaad6df8120bf3f7a68a8/dump.gsd --frame-start 32000 --frame-end 32401 \
		LoadModel -f "$<" --subsample 5e-4 \
		EvaluateEmbedding --average-bonds 0 \
		UMAPEmbedding \
		EmbeddingPlotter -r 1 --shuffle 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f cI2_W "$<" umap

output.EmbeddingPlotter.cP1_Po.%.pca.pdf: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${ZOOMED_DATA_DIR}/ae0780dec9c965ec0d7275b05153d380/dump.gsd --frame-start 54580 --frame-end 55631 --frame-skip 2 \
		LoadModel -f "$<" --subsample 5e-4 \
		EvaluateEmbedding --average-bonds 0 \
		PCAEmbedding \
		EmbeddingPlotter -r 1 --shuffle 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f cP1_Po "$<" pca

output.EmbeddingPlotter.cP1_Po.%.umap.pdf: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${ZOOMED_DATA_DIR}/ae0780dec9c965ec0d7275b05153d380/dump.gsd --frame-start 54580 --frame-end 55631 --frame-skip 2 \
		LoadModel -f "$<" --subsample 5e-4 \
		EvaluateEmbedding --average-bonds 0 \
		UMAPEmbedding \
		EmbeddingPlotter -r 1 --shuffle 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f cP1_Po "$<" umap

.PHONY: embedding_pdfs
embedding_pdfs:
	for f in ./dump.*;do ${PYTHON} -m flowws.run InitializeTF FileLoader -f $$(find ${DATA_DIR} -name dump.gsd | sort) --frame-start 5 --frame-skip 4 LoadModel -f $$f --subsample 5e-4 EvaluateEmbedding --average-bonds 0 PCAEmbedding EmbeddingPlotter --shuffle 1 flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f $$(basename $$f) pca ;done
	for f in ./dump.*;do ${PYTHON} -m flowws.run InitializeTF FileLoader -f $$(find ${DATA_DIR} -name dump.gsd | sort) --frame-start 5 --frame-skip 4 LoadModel -f $$f --subsample 5e-4 EvaluateEmbedding --average-bonds 0 UMAPEmbedding -n 64 EmbeddingPlotter --shuffle 1 flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f $$(basename $$f) umap ;done

output.ClassifierPlotter.%.pdf: %
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f $$(find ${DATA_DIR} -name dump.gsd | sort) --frame-start 5 --frame-skip 20 \
		LoadModel -f $< --subsample 5e-4 \
		ClassifierPlotter \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f $<

dump.frame_classification.cF4_traj.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		PermanentDropout \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/ff1e2b1da3eccc022eca819dd7108606/dump.gsd --frame-start 4 --frame-skip 4 \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${FRAME_CLASSIFICATION} --subsample .2 ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f frame_classification cF4_traj

output.ClassifierPlotter.cF4_traj.pdf: dump.frame_classification.cF4_traj.sqlite
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/ff1e2b1da3eccc022eca819dd7108606/dump.gsd --frame-start 5 --frame-skip 4 \
		LoadModel -f $< --subsample .2 \
		ClassifierPlotter \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f cF4_traj

dump.frame_classification.cI2_traj.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		PermanentDropout \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/81a6c5396a501f7e939001ffff307462/dump.gsd --frame-start 4 --frame-skip 4 \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${FRAME_CLASSIFICATION} --subsample .2 ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f frame_classification cI2_traj

output.ClassifierPlotter.cI2_traj.pdf: dump.frame_classification.cI2_traj.sqlite
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/81a6c5396a501f7e939001ffff307462/dump.gsd --frame-start 5 --frame-skip 4 \
		LoadModel -f $< --subsample .2 \
		ClassifierPlotter \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f cI2_traj

dump.frame_classification.hP2_traj.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		PermanentDropout \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/583e02d7d1ae5e2bba602747e2bf034d/dump.gsd --frame-start 4 --frame-skip 4 \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${FRAME_CLASSIFICATION} --subsample .2 ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f frame_classification hP2_traj

output.ClassifierPlotter.hP2_traj.pdf: dump.frame_classification.hP2_traj.sqlite
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/583e02d7d1ae5e2bba602747e2bf034d/dump.gsd --frame-start 5 --frame-skip 4 \
		LoadModel -f $< --subsample .2 \
		ClassifierPlotter \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f hP2_traj

dump.frame_classification.tP30_traj.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		PermanentDropout \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/e8ecb61993395e7ced88692e33fc3932/dump.gsd --frame-start 4 --frame-skip 4 \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${FRAME_CLASSIFICATION} --subsample .2 ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f frame_classification tP30_traj

output.ClassifierPlotter.tP30_traj.pdf: dump.frame_classification.tP30_traj.sqlite
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/e8ecb61993395e7ced88692e33fc3932/dump.gsd --frame-start 5 --frame-skip 4 \
		LoadModel -f $< --subsample .2 \
		ClassifierPlotter \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f tP30_traj

define SUMMARIZE_PYSCRIPT
import base64
import subprocess
import sys

lines = []
lines.append("""
<style>
body {
	background-color: #DDDDE2;
}
img {
	display: block;
	margin: 1em auto;
	max-width: 100%;
	border-radius: 1%;
}
p {
	text-align: center;
}
</style>
""")
for fname in sorted(sys.argv[2:]):
    command = ['gs', '-q', '-dSAFER', '-r72', '-sDEVICE=pngalpha', '-o', '-', fname]
    # command = ['convert', '-density', '72', fname, 'png:-']
    png_contents = subprocess.check_output(command)
    b64 = base64.b64encode(png_contents)
    lines.append('<p>{}</p>'.format(fname))
    lines.append('<img src="data:image/png;base64,{}" />'.format(b64.decode()))
    print(fname)

with open(sys.argv[1], 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
endef
export SUMMARIZE_PYSCRIPT
SUMMARIZE_SCRIPT := ${PYTHON} -c "$$SUMMARIZE_PYSCRIPT"

summary.%.html:
	$(SUMMARIZE_SCRIPT) "$@" output.EmbeddingPlotter.*.pdf

dump.self_classifier.%.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		PermanentDropout \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/$*/dump.gsd --frame-start 5 --frame-skip 4 \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${FRAME_CLASSIFICATION} -z 0 --subsample .1 LimitAccuracyCallback -c 0.333 $(subst --normalize-distances ${NET_DISTANCE_NORMALIZATION},--normalize-distances mean,${FRAME_CLASSIFICATION_ARCH}) ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f self_classifier $*

output.ClassifierPlotter.self_classifier.%.pdf: dump.self_classifier.%.sqlite
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/$*/dump.gsd --frame-start 6 --frame-skip 4 \
		LoadModel -f $< --subsample .1 \
		ClassifierPlotter --aggregate-probabilities 1 \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f self_classifier $*

summary.all_file_self_classifiers.html:
	for wid in $$(find ${DATA_DIR} | awk '/dump.gsd/{n=split($$0, bits, "/");print(bits[n-1])}'); do \
		$(MAKE) -f "$(THIS_FILE)" output.ClassifierPlotter.self_classifier.$${wid}.pdf; \
		done
	$(SUMMARIZE_SCRIPT) "$@" output.ClassifierPlotter.*.pdf

dump.self_regressor.%.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		PermanentDropout \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/$*/dump.gsd --frame-start 5 --frame-skip 4 \
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${FRAME_REGRESSION} --subsample .1 ${FRAME_REGRESSION_ARCH} ${FRAME_REGRESSION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f self_regressor $*

output.RegressorPlotter.self_regressor.%.pdf: dump.self_regressor.%.sqlite
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/$*/dump.gsd --frame-start 6 --frame-skip 4 \
		LoadModel -f $< --subsample .1 \
		RegressorPlotter \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f self_regressor $*

summary.all_file_self_regressors.html:
	for wid in $$(find ${DATA_DIR} | awk '/dump.gsd/{n=split($$0, bits, "/");print(bits[n-1])}'); do \
		$(MAKE) -f "$(THIS_FILE)" output.RegressorPlotter.self_regressor.$${wid}.pdf; \
		done
	$(SUMMARIZE_SCRIPT) "$@" output.RegressorPlotter.*.pdf

dump.phase_classifier.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		PermanentDropout \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/0dafc8d761ee6decf390e5aea3e3ab59/dump.gsd --frame-start 7 --frame-end 17 --custom-context phase 0 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/0dafc8d761ee6decf390e5aea3e3ab59/dump.gsd --frame-start 48 --frame-end 58 --custom-context phase 1 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/0dafc8d761ee6decf390e5aea3e3ab59/dump.gsd --frame-start 87 --frame-end 97 --custom-context phase 2 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/983ca2bd9b76acb76d304f02775010f6/dump.gsd --frame-start 5 --frame-end 15 --custom-context phase 0 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/983ca2bd9b76acb76d304f02775010f6/dump.gsd --frame-start 37 --frame-end 47 --custom-context phase 1 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/983ca2bd9b76acb76d304f02775010f6/dump.gsd --frame-start 77 --frame-end 87 --custom-context phase 2 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/c6705527d5f58167e12517e704645ad4/dump.gsd --frame-start 7 --frame-end 17 --custom-context phase 0 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/c6705527d5f58167e12517e704645ad4/dump.gsd --frame-start 48 --frame-end 58 --custom-context phase 1 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/c6705527d5f58167e12517e704645ad4/dump.gsd --frame-start 86 --frame-end 96 --custom-context phase 2 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/cd1b523e0e660c32f5cabf4b8acbcb46/dump.gsd --frame-start 9 --frame-end 19 --custom-context phase 0 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/cd1b523e0e660c32f5cabf4b8acbcb46/dump.gsd --frame-start 51 --frame-end 61 --custom-context phase 1 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/cd1b523e0e660c32f5cabf4b8acbcb46/dump.gsd --frame-start 88 --frame-end 98 --custom-context phase 2 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/f74adc62c9773ed4edeeb15681d4c20c/dump.gsd --frame-start 12 --frame-end 22 --custom-context phase 0 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/f74adc62c9773ed4edeeb15681d4c20c/dump.gsd --frame-start 48 --frame-end 58 --custom-context phase 1 \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/f74adc62c9773ed4edeeb15681d4c20c/dump.gsd --frame-start 82 --frame-end 92 --custom-context phase 2	\
		${NEIGHBOR_CALC} \
		NormalizeNeighborDistance --mode min \
		${FRAME_CLASSIFICATION} --subsample .1 ${FRAME_CLASSIFICATION_ARCH} ${FRAME_CLASSIFICATION_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f phase_classifier

output.ClassifierPlotter.phase_classifier.%.pdf: dump.phase_classifier.sqlite
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		FileLoader -f ${DATA_DIR}/struc_prediction/workspace/$*/dump.gsd --frame-start 6 --frame-skip 4 \
		LoadModel -f $< --subsample .1 \
		ClassifierPlotter \
		flowws_analysis.Save --matplotlib-figure-kwargs figsize 12,6 -f phase_classifier $*

summary.all_file_phase_classifiers.html:
	for wid in $$(find ${DATA_DIR} | awk '/dump.gsd/{n=split($$0, bits, "/");print(bits[n-1])}'); do \
		$(MAKE) -f "$(THIS_FILE)" output.ClassifierPlotter.phase_classifier.$${wid}.pdf; \
		done
	$(SUMMARIZE_SCRIPT) "$@" output.ClassifierPlotter.*.pdf

MOVIE_FRAME_DIR:=/tmp/frames

define MOVIE_PYSCRIPT
import vispy.app;vispy.app.use_app('pyglet')
import plato.draw.vispy as draw
import flowws
from flowws_analysis import *
from flowws_freud import *
from flowws_freud.SmoothBOD import optimize_rotation
import gtar
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as pp
import sys
import os
import rowan

if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

fname = sys.argv[1]
thresh = float(sys.argv[2]) if len(sys.argv) > 2 else None

with gtar.GTAR(fname, 'r') as traj:
    (_, frames) = traj.framesWithRecordsNamed('position')

n_frames = len(frames)
w = flowws.Workflow([
    GTAR(filename=fname),
    SmoothBOD(),
    Center(),
    Plato(additive_rendering=1, color_scale=.05),
])

orientation = None
optimize_kwargs = dict(lmax=12, rotations=128, patience=128, max_steps=1024)

scenes = []
orders = []
done_optimizing = False
for i in tqdm(range(n_frames)):
    w.stages[0].arguments['frame'] = n_frames - i - 1
    scope = w.run()

    if orientation is not None:
        rotation = rowan.mapping.kabsch(last_positions, scope['position'])[0]
        rotation = rowan.from_matrix(rotation, False)
        rotation = rowan.conjugate(rotation)
        orientation = rowan.multiply(orientation, rotation)
        orientation /= np.linalg.norm(orientation)
        optimize_kwargs['initial_rotation'] = orientation
        optimize_kwargs['max_steps'] = 128
        optimize_kwargs['theta_min'] = 1e-5
        optimize_kwargs['theta_max'] = np.pi/8 # /optimize_kwargs['max_steps']
        optimize_kwargs['mode'] = 'local'
        optimize_kwargs['patience'] = 4
        optimize_kwargs['rotations'] = 32
        optimize_kwargs['sph_index'] = np.argmax(first_opt.sphs.real)
    else:
        pass
    opt = optimize_rotation(scope['SmoothBOD.bonds'], **optimize_kwargs)
    orders.append(opt.op)

    if thresh is not None:
        done_optimizing |= opt.op < thresh
    if orientation is None:
        first_opt = opt
        first_orientation = opt.q
        first_positions = scope['position']
    if not done_optimizing:
        orientation = opt.q

    last_positions = scope['position']
    scene = scope['visuals'][1].draw_plato()
    scene.rotation = orientation
    scene.zoom = 3.25
    scene.size_pixels = 1080
    scenes.append(scene)

pp.plot(orders[::-1])
pp.xlabel('Frame'); pp.ylabel('Order')
if thresh is not None:
    pp.hlines(thresh, 0, len(orders), color='black', lw=3)
pp.savefig(os.path.join('${MOVIE_FRAME_DIR}', 'order.svg'))

scene = scenes[0].convert(draw)
scene.enable('static')

for i, ref_scene in tqdm(enumerate(reversed(scenes)), total=n_frames):
    for (src, dst) in zip(ref_scene, scene):
        dst.copy_from(src, True)
    scene.rotation = ref_scene.rotation
    target = os.path.join('${MOVIE_FRAME_DIR}', 'frame.{:05d}.png'.format(i))
    scene.save(target)
endef


.PHONY: all_pyriodic_leave_one_out all_noisy_bond_pyriodic_leave_one_out all_nearest_bond_pyriodic_leave_one_out all_autoencoder_pyriodic_leave_one_out all_denoising_pyriodic_leave_one_out all_shift_identification_pyriodic_leave_one_out all_frame_classification_pyriodic_leave_one_out
all_pyriodic_leave_one_out: all_noisy_bond_pyriodic_leave_one_out all_nearest_bond_pyriodic_leave_one_out all_autoencoder_pyriodic_leave_one_out all_denoising_pyriodic_leave_one_out all_shift_identification_pyriodic_leave_one_out all_frame_classification_pyriodic_leave_one_out
all_noisy_bond_pyriodic_leave_one_out: $(foreach struc,${PYRIODIC_ALL_STRUCTURES},dump.noisy_bond.pyriodic_leave_one_out.${struc}.sqlite)
all_denoising_pyriodic_leave_one_out: $(foreach struc,${PYRIODIC_ALL_STRUCTURES},dump.denoising.pyriodic_leave_one_out.${struc}.sqlite)

dump.noisy_bond.pyriodic_leave_one_out.%.sqlite:
	${PYTHON} -m flowws.run \
		InitializeTF ${INITIALIZETF_EXTRA_ARGS} \
		$(patsubst $(@:dump.noisy_bond.pyriodic_leave_one_out.%.sqlite=%),,${PYRIODIC_ALL_FRAMES}) \
		${NEIGHBOR_CALC} \
		${NOISY_BOND} ${NOISY_BOND_ARCH} ${NOISY_BOND_TRAIN} \
		flowws_keras_experimental.Save --save-model 1 -f noisy_bond pyriodic_leave_one_out $(@:dump.noisy_bond.pyriodic_leave_one_out.%.sqlite=%)

.SECONDARY:
