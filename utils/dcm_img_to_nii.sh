#!bin/bash
# Convert dicom-like images to nii files in 3D
# This is the first step for image pre-processing

# Feed path to the downloaded data here
DATAPATH=./data/CHAOST2/MR # please put chaos dataset training fold here which contains ground truth

# Feed path to the output folder here
OUTPATH=./data/CHAOST2/niis


if [ ! -d  $OUTPATH/T2SPIR ]
then
    mkdir $OUTPATH/T2SPIR
fi

# In order for the following commands to work, you need to install `dcm2niix` using pip, run: `pip install dcm2niix`

for sid in $(ls "$DATAPATH")
do
	dcm2niix -o "$DATAPATH/$sid/T2SPIR" "$DATAPATH/$sid/T2SPIR/DICOM_anon";
	find "$DATAPATH/$sid/T2SPIR" -name "*.nii" -exec mv {} "$OUTPATH/T2SPIR/image_$sid.nii" \;
done;


