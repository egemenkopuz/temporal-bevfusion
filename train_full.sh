#!/bin/bash

N_GPUS=2
PRETRAINED_INTER_LIDAR="checkpoints/tumtraf-i/inter-lidar_l-voxelnet-1600z51-0xy1-0z2-0st0-gtp15-aug3d-0sx9-1sy1--0rx2-0ry2-0t25-dtrmnstc/latest.pth"

for i in {1..4}
    # attach i to the auto run dir
    do
        AUTO_RUN_DIR="checkpoints/tumtraf-i-iter/$i"
        echo $AUTO_RUN_DIR
        # # sameaugall - gtp
        # torchpack dist-run -np $N_GPUS python tools/train.py \
        #     configs/tumtraf-i/temporal/transfusion/lidar/voxelnet-convlstm-1600g-0xy1-0z20-sameaugall-ql3-qrt1-gtp3-lfrz.yaml \
        #     --load_from $PRETRAINED_INTER_LIDAR \
        #     --auto-run-dir $AUTO_RUN_DIR; \
        # # sameaugall - gtp sameaugseq
        # torchpack dist-run -np $N_GPUS python tools/train.py \
        #     configs/tumtraf-i/temporal/transfusion/lidar/voxelnet-convlstm-1600g-0xy1-0z20-sameaugall-ql3-qrt1-gtp3-sameaug-lfrz.yaml \
        #     --load_from $PRETRAINED_INTER_LIDAR \
        #     --auto-run-dir $AUTO_RUN_DIR; \
        # # sameaugall - gtp sameaugseq trans
        # torchpack dist-run -np $N_GPUS python tools/train.py \
        #     configs/tumtraf-i/temporal/transfusion/lidar/voxelnet-convlstm-1600g-0xy1-0z20-sameaugall-ql3-qrt1-gtp3-sameaug-trans-lfrz.yaml \
        #     --load_from $PRETRAINED_INTER_LIDAR \
        #     --auto-run-dir $AUTO_RUN_DIR; \
        # # gtp sameaugseq trans rot
        # torchpack dist-run -np $N_GPUS python tools/train.py \
        #     configs/tumtraf-i/temporal/transfusion/lidar/voxelnet-convlstm-1600g-0xy1-0z20-ql3-qrt1-gtp3-sameaug-trans-rot-lfrz.yaml \
        #     --load_from $PRETRAINED_INTER_LIDAR \
        #     --auto-run-dir $AUTO_RUN_DIR; \
        # sameaugall - gtp sameaugseq trans rot but ql 2
        # torchpack dist-run -np $N_GPUS python tools/train.py \
        #     configs/tumtraf-i/temporal/transfusion/lidar/voxelnet-convlstm-1600g-0xy1-0z20-sameaugall-ql2-qrt1-gtp3-sameaug-trans-rot-lfrz.yaml \
        #     --load_from $PRETRAINED_INTER_LIDAR \
        #     --auto-run-dir $AUTO_RUN_DIR; \
        # # sameaugall - gtp sameaugseq trans rot but ql 4
        # torchpack dist-run -np $N_GPUS python tools/train.py \
        #     configs/tumtraf-i/temporal/transfusion/lidar/voxelnet-convlstm-1600g-0xy1-0z20-sameaugall-ql4-qrt1-gtp3-sameaug-trans-rot-lfrz.yaml \
        #     --load_from $PRETRAINED_INTER_LIDAR \
        #     --auto-run-dir $AUTO_RUN_DIR; \
        # # sameaugall - gtp sameaugseq trans rot but qrt 2
        # torchpack dist-run -np $N_GPUS python tools/train.py \
        #     configs/tumtraf-i/temporal/transfusion/lidar/voxelnet-convlstm-1600g-0xy1-0z20-sameaugall-ql3-qrt2-gtp3-sameaug-trans-rot-lfrz.yaml \
        #     --load_from $PRETRAINED_INTER_LIDAR \
        #     --auto-run-dir $AUTO_RUN_DIR; \
        # # sameaugall - gtp sameaugseq trans rot but qrt 0
        # torchpack dist-run -np $N_GPUS python tools/train.py \
        #     configs/tumtraf-i/temporal/transfusion/lidar/voxelnet-convlstm-1600g-0xy1-0z20-sameaugall-ql3-qrt0-gtp3-sameaug-trans-rot-lfrz.yaml \
        #     --load_from $PRETRAINED_INTER_LIDAR \
        #     --auto-run-dir $AUTO_RUN_DIR; \

        # sameaugall - gtp sameaugseq trans rot rpd 0.5
        torchpack dist-run -np $N_GPUS python tools/train.py \
            configs/tumtraf-i/temporal/transfusion/lidar/voxelnet-convlstm-1600g-0xy1-0z20-sameaugall-ql3-qrt1-gtp3-sameaug-rpd0p5-trans-rot-lfrz.yaml \
            --load_from $PRETRAINED_INTER_LIDAR \
            --auto-run-dir $AUTO_RUN_DIR; \

        # # sameaugall - gtp sameaugseq trans rot rb 16
        # torchpack dist-run -np $N_GPUS python tools/train.py \
        #     configs/tumtraf-i/temporal/transfusion/lidar/voxelnet-convlstm-1600g-0xy1-0z20-rb16-sameaugall-ql3-qrt1-gtp3-sameaug-trans-rot-lfrz.yaml \
        #     --load_from $PRETRAINED_INTER_LIDAR \
        #     --auto-run-dir $AUTO_RUN_DIR; \
        # # sameaugall - gtp sameaugseq trans rot rb 4
        # torchpack dist-run -np $N_GPUS python tools/train.py \
        #     configs/tumtraf-i/temporal/transfusion/lidar/voxelnet-convlstm-1600g-0xy1-0z20-rb4-sameaugall-ql3-qrt1-gtp3-sameaug-trans-rot-lfrz.yaml \
        #     --load_from $PRETRAINED_INTER_LIDAR \
        #     --auto-run-dir $AUTO_RUN_DIR; \

    done
