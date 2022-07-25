#!/bin/bash

export slow5tools=$HOME/DeepSelectNet/tools/slow5tools-v0.3.0/slow5tools
export minimap2=$HOME/DeepSelectNet/tools/minimap2/minimap2

POS_SLOW5=$1
NEG_SLOW5=$2
POS_FASTQ=$3
NEG_FASTQ=$4
POS_REF=$5
NEG_REF=$6
BASE_LENGTH=$7
POS_READ_IDS='read-ids-pos.txt'
NEG_READ_IDS='read-ids-neg.txt'
MIXED_REF='mixed-ref.fasta'
POS_CONTIGS='pos-contigs.list'
NEG_CONTIGS='neg-contigs.list'
TMP_POS_FASTQ='pos-fastq.fastq'
TMP_NEG_FASTQ='neg-fastq.fastq'
MAPPED_POS='mapped-pos'$BASE_LENGTH'.paf'
MAPPED_NEG='mapped-neg'$BASE_LENGTH'.paf'
TRUE_POS='true-pos'$BASE_LENGTH'.txt'
TRUE_NEG='true-neg'$BASE_LENGTH'.txt'
FALSE_POS='false-pos'$BASE_LENGTH'.txt'
FALSE_NEG='false-neg'$BASE_LENGTH'.txt'

if [ ! -x $slow5tools ]; then
  echo "slow5tools does not exists in provided directory. Please provide correct path as -> export slow5tools=<path_slow5tools>" && exit
fi

if [ ! -x "$minimap2" ]; then
  echo "minimap2 does not exists in provided directory. Please provide correct path as -> export minimap2=<path_minimap2>" && exit
fi

if test -f "$POS_READ_IDS"; then
  echo "$POS_READ_IDS already exists in directory."
else
  echo "Extracting pos read ids."
  $slow5tools view $POS_SLOW5 | grep -v '^[#@]' | awk '{print $1}' > $POS_READ_IDS
  #python ~/DeepSelectNet/support/export_read_id.py -f5 $POS_FAST5_PATH -o $POS_READ_IDS
fi

if test -f "$NEG_READ_IDS"; then
  echo "$NEG_READ_IDS already exists in directory."
else
  echo "Extracting neg read ids."
  $slow5tools view $NEG_SLOW5 | grep -v '^[#@]' | awk '{print $1}' > $NEG_READ_IDS
  #python ~/DeepSelectNet/support/export_read_id.py -f5 $NEG_FAST5_PATH -o $NEG_READ_IDS
fi

cat $POS_REF $NEG_REF > $MIXED_REF

if test -z "$BASE_LENGTH"; then
  echo "Using full read for minimap2"
  cat $POS_FASTQ > $TMP_POS_FASTQ
  cat $NEG_FASTQ > $TMP_NEG_FASTQ
else
  echo "Using $BASE_LENGTH bases read for minimap2"
  awk -v N="${BASE_LENGTH}" '{if(NR%2==1){print $0} else{print substr($1,1,N)}}' $POS_FASTQ > $TMP_POS_FASTQ
  awk -v N="${BASE_LENGTH}" '{if(NR%2==1){print $0} else{print substr($1,1,N)}}' $NEG_FASTQ > $TMP_NEG_FASTQ
fi

if test -f $MAPPED_POS;then
  echo "Minimap for positive set is already completed."
else
  echo "Minimap for positive set is starting..."
  $minimap2 -cx map-ont --secondary=no $MIXED_REF $TMP_POS_FASTQ > $MAPPED_POS
  echo "Minimap for positive set is completed."
fi

if test -f $MAPPED_NEG;then
  echo "Minimap for negative set is already completed."
else
  echo "Minimap for negative set is starting..."
  $minimap2 -cx map-ont --secondary=no $MIXED_REF $TMP_NEG_FASTQ > $MAPPED_NEG
  echo "Minimap for negative set is completed."
fi

cat $POS_REF | grep "^>" | awk '{print $1}' | cut -c 2- | sort -u > $POS_CONTIGS
#cat $NEG_REF | grep "^>" | awk '{print $1}' | cut -c 2- | sort -u > $NEG_CONTIGS

grep -w -F -f $POS_CONTIGS $MAPPED_POS | awk '{print $1}' > $TRUE_POS
grep -w -F -f $NEG_CONTIGS $MAPPED_POS | awk '{print $1}' > $FALSE_POS
grep -w -F -f $NEG_CONTIGS $MAPPED_NEG | awk '{print $1}' > $TRUE_NEG
grep -w -F -f $POS_CONTIGS $MAPPED_NEG | awk '{print $1}' > $FALSE_NEG

TP=$(cat $TRUE_POS | sort -u | wc -l)
TN=$(cat $TRUE_NEG | sort -u | wc -l)

FP=$(cat $FALSE_POS | sort -u | wc -l)
FN=$(cat $FALSE_NEG | sort -u | wc -l)


echo "True positive reads: $TP"
echo "False positive reads: $FP"
echo "True negative reads: $TN"
echo "False negative reads: $FN"

rm $MIXED_REF
rm $POS_CONTIGS
#rm $NEG_CONTIGS
rm $TMP_POS_FASTQ
rm $TMP_NEG_FASTQ
rm $MAPPED_POS
rm $MAPPED_NEG
rm $TRUE_POS
rm $TRUE_NEG
rm $FALSE_POS
rm $FALSE_NEG