#!/bin/bash

POS_FAST5_PATH=$1
NEG_FAST5_PATH=$2
POS_FASTQ_PATH=$3
NEG_FASTQ_PATH=$4
MIXED_REF=$5
BASE_LENGTH=$6
POS_READ_IDS='read-ids-pos.txt'
NEG_READ_IDS='read-ids-neg.txt'
TEST_FASTQ='testset'$BASE_LENGTH'.fastq'
MAPPED_NAME='mapped'$BASE_LENGTH'.paf'
MAPPED_POS='mapped-pos'$BASE_LENGTH'.txt'
MAPPED_NEG='mapped-neg'$BASE_LENGTH'.txt'
TRUE_POS='true-pos'$BASE_LENGTH'.txt'
TRUE_NEG='true-neg'$BASE_LENGTH'.txt'


if test -f "$POS_READ_IDS"; then
  echo "$POS_READ_IDS already exists in directory."
else
  echo "Extracting pos read ids."
  python ~/DeepSelectNet/support/export_read_id.py -f5 $POS_FAST5_PATH -o $POS_READ_IDS
fi

if test -f "$NEG_READ_IDS"; then
  echo "$NEG_READ_IDS already exists in directory."
else
  echo "Extracting neg read ids."
  python ~/DeepSelectNet/support/export_read_id.py -f5 $NEG_FAST5_PATH -o $NEG_READ_IDS
fi

cat $POS_FASTQ_PATH $NEG_FASTQ_PATH > TEST_FASTQ
python ~/DeepSelectNet/support/fastq_trimmer.py -fq TEST_FASTQ


if test -f $MAPPED_NAME;then
  echo "Minimap is already completed."
else
  echo "Minimap is starting..."
  ~/DeepSelectNet/tools/minimap2/minimap2 -cx map-ont --secondary=no $MIXED_REF TEST_FASTQ > $MAPPED_NAME
  echo "Minimap is completed."
fi



cat $MAPPED_NAME | awk '{if($6!~/N/) print $1}' | sort -u > $MAPPED_POS
cat $MAPPED_NAME | awk '{if($6~/N/) print $1}' | sort -u > $MAPPED_NEG

POS_TOTAL=$(cat $MAPPED_POS| wc -l)
NEG_TOTAL=$(cat $MAPPED_NEG| wc -l)


if test -f $MAPPED_POS;then
   grep -Ff $MAPPED_POS $POS_READ_IDS > $TRUE_POS
else
  echo "Skipping accuracy calculations since positive read filtering has failed."
fi


if test -f $MAPPED_NEG;then
   grep -Ff $MAPPED_NEG $NEG_READ_IDS > $TRUE_NEG
else
  echo "Skipping accuracy calculations since negative read filtering has failed."
fi

TP=$(cat $TRUE_POS | sort -u | wc -l)
TN=$(cat $TRUE_NEG | sort -u | wc -l)

FP=$(expr $POS_TOTAL - $TP)
FN=$(expr $NEG_TOTAL - $TN)

echo "True positive reads: $TP"
echo "False positive reads: $FP"
echo "True negative reads: $TN"
echo "False negative reads: $FN"

