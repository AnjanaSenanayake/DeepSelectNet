BLOW5_FILE=$1
TRIM_LEN=$2

export slow5tools=$HOME/DeepSelectNet/tools/slow5tools-v0.3.0/slow5tools

if [ ! -x $slow5tools ]; then
  echo "slow5tools does not exists in provided directory. Please provide correct path as -> export slow5tools=<path_slow5tools>" && exit
fi

echo "Start trimming Slow5 reads for $TRIM_LEN raw signals..."

echo "Exporting headers..."
$slow5tools view $BLOW5_FILE | grep '^[@#]' > headers.tmp

echo "Exporting eligible reads..."
$slow5tools view $BLOW5_FILE | grep -v '^[@#]' | awk '{if($7>=4500) print$0}' > eligible-reads.tmp

echo "Trimming started for exported reads..."
cat eligible-reads.tmp | cut -f 1-6 > eligible-reads-col1-6.tmp
cat eligible-reads-col1-6.tmp | awk -v trim_len="${TRIM_LEN}" '{print$0"\t"trim_len}' > eligible-reads-col1-7.tmp
cat eligible-reads.tmp |  cut -f 8 | cut -d "," -f -$TRIM_LEN > eligible-reads-col8-trimmed.tmp
cat eligible-reads.tmp |  cut -f 9- > eligible-reads-col9-.tmp
paste eligible-reads-col1-7.tmp eligible-reads-col8-trimmed.tmp eligible-reads-col9-.tmp > preprocessed-reads.tmp
paste eligible-reads-col1-7.tmp eligible-reads-col8-trimmed.tmp eligible-reads-col9-.tmp > preprocessed-reads.tmp
cat headers.tmp preprocessed-reads.tmp > preprocessed-reads.slow5
echo "Trimming completed for exported reads..."

echo "Removing temporary files..."
rm *.tmp
echo "Slow5 reads are trimmed successfully!"