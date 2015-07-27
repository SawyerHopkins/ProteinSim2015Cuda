echo Making core library
cd ClusteredCore
make clean
./ClusteredCore.sh
cd ../

echo Making user interface
cd ClusteredUI
make clean
cp ../ClusteredCore/bin/libcore.a linked/libcore.a
./ClusteredUI.sh
cd ../