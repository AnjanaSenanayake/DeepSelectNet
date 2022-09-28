BLOW5_FILE=$1
READ_LEN=$2
TRAIN_SIZE=$3
TEST_SIZE=$4
PREFIX=$5

export slow5tools=$HOME/DeepSelectNet/tools/slow5tools-v0.3.0/slow5tools

if [ ! -x "$slow5tools" ]; then
  echo "slow5tools does not exists in provided directory. Please provide correct path as -> export slow5tools=<path_slow5tools>" && exit
fi

echo "Processing is started..."
$slow5tools view "$BLOW5_FILE" | grep -v '^[#@]' | awk -v LEN="${READ_LEN}" '{if($7>LEN) print $1}' > 'read_ids_filtered.tmp'

NO_OF_READS_AVAILABLE=$(cat 'read_ids_filtered.tmp' | sort -u | wc -l)
NO_OF_READS_REQUIRED=$(($TRAIN_SIZE + $TEST_SIZE))

echo 'Available Reads: ' $NO_OF_READS_AVAILABLE
echo 'Required Reads: ' $NO_OF_READS_REQUIRED

if [ "$NO_OF_READS_AVAILABLE" -ge "$NO_OF_READS_REQUIRED" ]; then
  echo "Available reads are enough to meet the partition sizes"
else
  echo "Available reads are not enough to meet the partition sizes"
  echo "Aborting..."
  exit 0
fi

echo "Partitioning started..."
head -$TRAIN_SIZE read_ids_filtered.tmp > 'read_ids_train.tmp'
tail -$TEST_SIZE read_ids_filtered.tmp > 'read_ids_test.tmp'

$slow5tools get $BLOW5_FILE -l 'read_ids_train.tmp' -o $PREFIX'-train-'$READ_LEN'.blow5'
$slow5tools get $BLOW5_FILE -l 'read_ids_test.tmp' -o $PREFIX'-test-'$READ_LEN'.blow5'

rm -r *.tmp
echo "Partitioning completed."