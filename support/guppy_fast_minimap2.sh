#!/bin/bash

export minimap2=$HOME/DeepSelectNet/tools/minimap2/minimap2

POS_FASTQ=$1
NEG_FASTQ=$2
POS_REF=$3
NEG_REF=$4

MIXED_REF='mixed-ref.fasta.tmp'
POS_CONTIGS='pos-contigs.tmp'
NEG_CONTIGS='neg-contigs.tmp'

MAPPED_POS='mapped-pos.paf.tmp'
MAPPED_NEG='mapped-neg.paf.tmp'
TRUE_POS='true-pos.list'
TRUE_NEG='true-neg.list'
FALSE_POS='false-pos.list'
FALSE_NEG='false-neg.list'


if [ ! -x "$minimap2" ]; then
  echo "minimap2 does not exists in provided directory. Please provide correct path as -> export minimap2=<path_minimap2>" && exit
fi

cat $POS_REF $NEG_REF > $MIXED_REF

if test -f $MAPPED_POS;then
  echo "Minimap for positive set is already completed."
else
  echo "Minimap for positive set is starting..."
  $minimap2 -x map-ont --secondary=no $MIXED_REF $POS_FASTQ > $MAPPED_POS
  echo "Minimap for positive set is completed."
fi

if test -f $MAPPED_NEG;then
  echo "Minimap for negative set is already completed."
else
  echo "Minimap for negative set is starting..."
  $minimap2 -x map-ont --secondary=no $MIXED_REF $NEG_FASTQ > $MAPPED_NEG
  echo "Minimap for negative set is completed."
fi

cat $POS_REF | grep "^>" | awk '{print $1}' | cut -c 2- | sort -u > $POS_CONTIGS
cat $NEG_REF | grep "^>" | awk '{print $1}' | cut -c 2- | sort -u > $NEG_CONTIGS

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


#rm '*.tmp'