input_file=$1
output_name=$2
is_save_caches=$3

echo "Start compling ..."

echo "1. Preprocessing .cpp to .i"
g++ -E $input_file -o $output_name.i
echo "2. Compiling .i to .s"
g++ -S $output_name.i -o $output_name.s
echo "3. Assembling .s to .o"
g++ -c $output_name.s -o $output_name.o
echo "4. Linking .o to .exe"
g++ $output_name.o -o $output_name

if [ "$is_save_caches" = "true" ]; then
    echo "Saving caches ..."
    mkdir -p ./.caches
    mv $output_name.i $output_name.s $output_name.o ./.caches/
else
    echo "Deleting caches ..."
    rm -f $output_name.i $output_name.s $output_name.o
fi

echo "Finished."
echo "---------------------"

./$output_name.exe