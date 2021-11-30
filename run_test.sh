#!/bin/bash
#for s in "white" "black" "asian" "indian" "others"
for s in "indian"
do
	for race in "white"
	#for race in "white"
	do
		echo "${s} to ${race}"
		CUDA_LAUNCH_BLOCKING=1 python3 inference_images.py --csv utk_face/only_${race}_test.csv \
			   	 	--graph train_model/model_Adam_L1Loss_LRDecay_weightDecay0.0001_batch50_lr0.0005_LB0.01_srcasian_trgwhite.pth   \
					--image_folder utk_face \
			    		> pred_dir/${s}_to_${race}.txt
	done
done
