# (mpiFCCpx utofu_pp_1to1.cpp -o utofu_pp -ltofucom) && (cp utofu_pp output1/) \
#     &&  (cp utofu_pp_pjsub output1/) && (cd output1 && pjsub -s utofu_pp_pjsub)

# (mpiFCCpx utofu_pp.cpp -o utofu_pp -ltofucom) && (cp utofu_pp output/) \
#     &&  (cp utofu_pp_pjsub output/) && (cd output && pjsub -s utofu_pp_pjsub)



if [ $# -eq 0 ] ; then
    echo "no input parameter"
    exit
fi

if [ ! -d output  ];then
  mkdir output
fi


# if [ "$1" = "clean" ] ; then
#     echo "clean all"
#     rm -rf output/*
#     exit
# fi

outfile="output/output_$1"

if [ ! -d $outfile  ];then
  mkdir $outfile
fi

echo "cd $outfile and run"

# (cp lmp_execute $outfile/) &&  (cp lmp_deepmd_pjsub $outfile/) &&  (cp water.lmp $outfile/) &&  (cp in.deepmd $outfile/) && \
#          (cd $outfile && pjsub --interact --sparam wait-time=600 lmp_deepmd_pjsub)
(cp lmp_execute_* $outfile/) &&  (cp lmp_deepmd_pjsub $outfile/) &&  (cp water.lmp $outfile/) &&  (cp in.deepmd_* $outfile/) && (cp pb_data_water.dat $outfile/) && \
         (cd $outfile && pjsub -s lmp_deepmd_pjsub)

